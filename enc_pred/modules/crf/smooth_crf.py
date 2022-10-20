import torch
from allennlp.modules.conditional_random_field import ConditionalRandomField
from allennlp.models.model import Model
from allennlp.nn.util import logsumexp
from overrides import overrides


@Model.register("smooth-crf")
class SmoothCRF(ConditionalRandomField):
    def forward(
        self, inputs: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor = None,
        aggregate: bool = True):
        """

        :param inputs: Shape [batch, token, tag]
        :param tags: Shape [batch, token] or [batch, token, tag]
        :param mask: Shape [batch, token]
        :return:
        """
        if mask is None:
            mask = tags.new_ones(tags.shape, dtype=torch.bool)
        mask = mask.to(dtype=torch.bool)
        if tags.dim() == 2:
            return super(SmoothCRF, self).forward(inputs, tags, mask)

        # smooth mode
        log_denominator = self._input_likelihood(inputs, mask)
        log_numerator = self._smooth_joint_likelihood(inputs, tags, mask)

        nll = log_numerator - log_denominator

        return torch.sum(nll) if aggregate else nll

    def _smooth_joint_likelihood(
        self, logits: torch.Tensor, soft_tags: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        batch_size, sequence_length, num_tags = logits.size()

        epsilon = 1e-30
        soft_tags = soft_tags.clone()
        soft_tags[soft_tags < epsilon] = epsilon

        # Transpose batch size and sequence dimensions
        mask = mask.transpose(0, 1).contiguous()
        logits = logits.transpose(0, 1).contiguous()
        soft_tags = soft_tags.transpose(0, 1).contiguous()

        # Initial alpha is the (batch_size, num_tags) tensor of likelihoods combining the
        # transitions to the initial states and the logits for the first timestep.
        if self.include_start_end_transitions:
            alpha = self.start_transitions.view(1, num_tags) + logits[0] + soft_tags[0].log()
        else:
            alpha = logits[0] * soft_tags[0]

        # For each i we compute logits for the transitions from timestep i-1 to timestep i.
        # We do so in a (batch_size, num_tags, num_tags) tensor where the axes are
        # (instance, current_tag, next_tag)
        for i in range(1, sequence_length):
            # The emit scores are for time i ("next_tag") so we broadcast along the current_tag axis.
            emit_scores = logits[i].view(batch_size, 1, num_tags)
            # Transition scores are (current_tag, next_tag) so we broadcast along the instance axis.
            transition_scores = self.transitions.view(1, num_tags, num_tags)
            # Alpha is for the current_tag, so we broadcast along the next_tag axis.
            broadcast_alpha = alpha.view(batch_size, num_tags, 1)

            # Add all the scores together and logexp over the current_tag axis.
            inner = broadcast_alpha + emit_scores + transition_scores + soft_tags[i].log().unsqueeze(1)

            # In valid positions (mask == True) we want to take the logsumexp over the current_tag dimension
            # of `inner`. Otherwise (mask == False) we want to retain the previous alpha.
            alpha = logsumexp(inner, 1) * mask[i].view(batch_size, 1) + alpha * (
                ~mask[i]
            ).view(batch_size, 1)

        # Every sequence needs to end with a transition to the stop_tag.
        if self.include_start_end_transitions:
            stops = alpha + self.end_transitions.view(1, num_tags)
        else:
            stops = alpha

        # Finally we log_sum_exp along the num_tags dim, result is (batch_size,)
        return logsumexp(stops)

    def marginalize(self, step_logits: torch.Tensor, step_masks: torch.Tensor) -> torch.Tensor:
        """This function is used to generate step logits
        that marginalize all positions.

        step_logits: [n_batch * n_size, n_tags, n_labels]
        step_masks: [n_batch * n_size, n_tags]
        """
        total_input_likelihood = self._input_likelihood(step_logits, step_masks)
        alpha = self._forward_pass(step_logits, mask=step_masks)
        beta = self._backward_pass(step_logits, mask=step_masks)

        # all_positionwise_logits = []

        # for logits, mask, iplkd in zip(
        #         step_logits.split(1, dim=0),
        #         step_masks.split(1, dim=0),
        #         total_input_likelihood.split(1, dim=0)):
        #     logits = logits.squeeze(0)
        #     mask = mask.squeeze(0)
        #     iplkd = iplkd.squeeze()
        #     indices = mask.nonzero(as_tuple=False).squeeze()
        #     logits = torch.index_select(
        #         logits, dim=0,
        #         index=indices
        #     )  # [sequence_length, n_labels]

        #     # marginalize out other positions.
        #     alpha = self._forward_pass(logits)
        #     beta = self._backward_pass(logits)

        #     positionwise_logits = torch.nn.functional.pad(
        #             alpha + beta - iplkd,
        #             pad=(0, 0, 0, step_masks.shape[-1] - indices.shape[0]),
        #             mode='constant',
        #             value=0.
        #     )  # [n_tags (max), n_labels]
        
        return alpha + beta - total_input_likelihood.view(-1, 1, 1)  # [batch_size, sequence_length, num_labels]

    def _forward_pass(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        logits: [batch_size, sequence_length, n_labels]
        mask: [batch_size, sequence_length]
        """

        # TODO: batchify this calculation

        batch_size, sequence_length, num_tags = logits.size()

        alphas = []

        if mask is None:
            mask = torch.ones((batch_size, sequence_length,), dtype=torch.bool, device=logits.device)

        # Transpose batch size and sequence dimensions
        mask = mask.transpose(0, 1).contiguous()
        logits = logits.transpose(0, 1).contiguous()


        if self.include_start_end_transitions:
            alpha = self.start_transitions.view(1, num_tags) + logits[0]
        else:
            alpha = logits[0]  # [batch_size, num_tags]

        alphas.append(alpha)

        for i in range(1, sequence_length):
            emit_scores = logits[i].view(batch_size, 1, num_tags)
            transition_scores = self.transitions.view(1, num_tags, num_tags)
            broadcast_alpha = alpha.view(batch_size, num_tags, 1)

            inner = broadcast_alpha + (emit_scores + transition_scores)
            alpha = logsumexp(inner, 1) * mask[i].view(batch_size, 1) + alpha * (
                ~mask[i]
            ).view(batch_size, 1)
            alphas.append(alpha)

        if self.include_start_end_transitions:
            stops = alpha + self.end_transitions.view(1, num_tags)
        else:
            stops = alpha

        return torch.stack(alphas, dim=1)

    def _backward_pass(self, logits: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        logits: [batch_size, sequence_length, n_labels]
        mask: [batch_size, sequence_length]
        """

        batch_size, sequence_length, num_tags = logits.size()
        betas = []

        if mask is None:
            mask = torch.ones((batch_size, sequence_length,), dtype=torch.bool, device=logits.device)
        
        mask = mask.transpose(0, 1).contiguous()
        logits = logits.transpose(0, 1).contiguous()  # [sequence_length, batch_size, num_labels]

        if self.include_start_end_transitions:
            beta = self.end_transitions.view(1, num_tags) + torch.zeros_like(logits[-1])
        else:
            beta = torch.zero_like(logits[-1])

        betas.append(beta)

        for i in range(sequence_length - 1, 0, -1):
            emit_scores = logits[i].view(batch_size, 1, num_tags)
            transition_scores = self.transitions.view(1, num_tags, num_tags)
            broadcast_beta = beta.view(batch_size, 1, num_tags)

            inner = broadcast_beta + (emit_scores + transition_scores)

            beta = logsumexp(inner, 2) * mask[i].view(batch_size, 1) + beta * (
                ~mask[i]
            ).view(batch_size, 1)
            betas.insert(0, beta)

        return torch.stack(betas, dim=1)
