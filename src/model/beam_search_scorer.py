import warnings
from collections import UserDict
from typing import Optional, Tuple, Union
import torch

class BeamSearchScorer:


    def __init__(
        self,
        batch_size: int,
        num_beams: int,
        device: torch.device,
        length_penalty: Optional[float] = 1.0,
        **kwargs,
    ):
        self.num_beams = num_beams
        self.device = device
        self.length_penalty = length_penalty
        self.num_beam_hyps_to_keep = num_beams

        self._is_init = False
        self._beam_hyps = [
            BeamHypotheses(
                num_beams=self.num_beams,
                length_penalty=self.length_penalty
            )
            for _ in range(batch_size)
        ]
        self._done = torch.tensor([False for _ in range(batch_size)], dtype=torch.bool, device=self.device)

        if not isinstance(num_beams, int) or num_beams <= 1:
            raise ValueError(
                f"`num_beams` has to be an integer strictly greater than 1, but is {num_beams}. For `num_beams` == 1, one should make use of `greedy_search` instead."
            )

    @property
    def is_done(self) -> bool:
        return self._done.all()

    def process(
        self,
        current_best_beam: Union[torch.Tensor, None],
        next_scores: torch.FloatTensor,
        next_labels: torch.LongTensor,
        next_comb_idx: torch.Tensor,
        best_previous_beam_indices: torch.LongTensor,
    ) -> UserDict:
        """

        :param current_best_beam:  (batch_size, 2 * beam_size, height, 4 )
        :param next_labels:  (batch_size, 2 * beam_size, 4 )
        :param next_comb: (batch_size, 2 * num_beams) ## combination id.
        :param scores: (batch_size , beam_size*2)
        :param best_previous_beam_indices: (batch_size, self.num_beams)
        :param device:
        :return:
            next_beam_scres: (batch_size, num_beams)
            next_beam_labels: (batch_size, num_beams, 4)
            next_best_comb (batch_size, num_beams)
        """
        cur_height = current_best_beam.shape[-2] if current_best_beam is not None else 0 # batch_size, height, 4 (left idx, right idx, label id, op idx)
        batch_size = len(self._beam_hyps)
        if current_best_beam is not None:
            assert batch_size == current_best_beam.shape[0]

        device = next_scores.device
        next_beam_scores = torch.zeros((batch_size, self.num_beams), dtype=next_scores.dtype, device=device)
        next_beam_labels = torch.zeros((batch_size, self.num_beams, 4), dtype=next_labels.dtype, device=device)
        next_beam_indices = torch.zeros((batch_size, self.num_beams), dtype=best_previous_beam_indices.dtype, device=device)
        next_beam_comb = torch.zeros((batch_size, self.num_beams), dtype=next_comb_idx.dtype, device=device)

        for batch_idx, beam_hyp in enumerate(self._beam_hyps):
            beam_hyp = self._beam_hyps[batch_idx]
            if self._done[batch_idx]:
                # assert (
                #     len(beam_hyp) >= self.num_beams
                # ), f"Batch can only be done if at least {self.num_beams} beams have been generated"
                next_beam_scores[batch_idx, :] = -10000
                next_beam_labels[batch_idx, :, :] = 0
                next_beam_comb[batch_idx, :] = 0
                next_beam_indices[batch_idx, :] = 0
                continue

            # next tokens for this sentence
            beam_idx = 0
            for beam_rank, (next_label, next_score, best_previous_beam_idx, best_next_comb_idx) in enumerate(
                zip(next_labels[batch_idx], next_scores[batch_idx], best_previous_beam_indices[batch_idx], next_comb_idx[batch_idx])
            ):
                if next_label[-1] == 1:  ## last_h, stop_id == 1
                    is_beam_token_worse_than_top_num_beams = beam_rank >= self.num_beams
                    if is_beam_token_worse_than_top_num_beams:
                        continue
                    ## check stop = 0 in previous labels
                    non_stop_next_label = next_label.clone()
                    non_stop_next_label[-1] = 0
                    idx = ((non_stop_next_label.unsqueeze(0).expand(next_labels[batch_idx].size(0), 4) == next_labels[batch_idx]).sum(dim=-1) == 4).nonzero()
                    if len(idx) != 0:
                        if idx[0,0] < beam_rank:
                            continue
                    if current_best_beam is not None:
                        beam_hyp.add(
                            torch.cat([current_best_beam[batch_idx, best_previous_beam_idx], next_label.unsqueeze(0)], dim=0).long(),
                            next_score.item(),
                        )
                    else:
                        beam_hyp.add(
                            next_label.unsqueeze(0).long(),
                            next_score.item(),
                        )

                    # next_beam_scores[batch_idx, beam_idx] = next_score
                    # next_beam_labels[batch_idx, beam_idx] = next_label
                    # next_beam_indices[batch_idx, beam_idx] = best_previous_beam_idx
                    # next_beam_comb[batch_idx, beam_idx] = best_next_comb_idx
                    # beam_idx += 1

                else:
                    # check stop = 1 in previous labels
                    stop_next_label = next_label.clone()
                    stop_next_label[-1] = 1
                    idx = ((stop_next_label.unsqueeze(0).expand(next_labels[batch_idx].size(0), 4) == next_labels[batch_idx]).sum(dim=-1) == 4).nonzero()
                    if len(idx) != 0:
                        if idx[0, 0] < beam_rank:
                            continue

                    # add next predicted token since it is not eos_token
                    next_beam_scores[batch_idx, beam_idx] = next_score
                    next_beam_labels[batch_idx, beam_idx] = next_label
                    next_beam_indices[batch_idx, beam_idx] = best_previous_beam_idx
                    next_beam_comb[batch_idx, beam_idx] = best_next_comb_idx
                    beam_idx += 1

                if beam_idx == self.num_beams:
                    break

            if beam_idx == 0:
                self._done[batch_idx] = True
            if beam_idx < self.num_beams:
                next_beam_scores[batch_idx, beam_idx:] = -100000
                # next_beam_labels[batch_idx, beam_idx:] = curr_labels
                # next_best_comb[batch_idx, beam_idx:] = best_comb[batch_idx, beam_rank]
            ## might be assertion error as well
            # assert beam_idx == self.num_beams
            # Check if we are done so that we can save a pad step if all(done)
            self._done[batch_idx] = self._done[batch_idx] or beam_hyp.is_done(
                next_beam_scores[batch_idx].max().item(), cur_height + 1
            )

        return UserDict(
            {
                "next_beam_scores": next_beam_scores,
                "next_beam_labels": next_beam_labels,
                "next_best_previous_beam_indices": next_beam_indices,
                "next_beam_comb_idx": next_beam_comb,
            }
        )

    def finalize(
        self,
        current_best_beam: Union[torch.Tensor, None],
        final_beam_scores: torch.FloatTensor,
        max_height: int,
    ) -> UserDict:
        batch_size = len(self._beam_hyps)

        # finalize all open beam hypotheses and add to generated hypotheses
        for batch_idx, beam_hyp in enumerate(self._beam_hyps):
            if self._done[batch_idx]:
                continue

            # all open beam hypotheses are added to the beam hypothesis
            # beam hypothesis class automatically keeps the best beams
            for beam_id in range(self.num_beams):
                final_score = final_beam_scores[batch_idx, beam_id].item()
                final_labels = current_best_beam[batch_idx, beam_id]
                beam_hyp.add(final_labels, final_score)

        # select the best hypotheses
        heights = current_best_beam.new(batch_size, self.num_beam_hyps_to_keep)
        batch_best = []
        best_scores = torch.zeros((batch_size, self.num_beam_hyps_to_keep), device=self.device, dtype=torch.float32)
        best_scores.fill_(-1000)

        # retrieve best hypotheses
        for i, beam_hyp in enumerate(self._beam_hyps):
            sorted_hyps = sorted(beam_hyp.beams, key=lambda x: x[0])
            best = []
            for j in range(min(self.num_beam_hyps_to_keep, len(sorted_hyps))):
                best_hyp_tuple = sorted_hyps.pop()
                best_score = best_hyp_tuple[0]
                best_hyp = best_hyp_tuple[1]
                heights[i, j] = len(best_hyp)

                # append to lists
                best.append(best_hyp)
                best_scores[i, j] = best_score
            batch_best.append(best)

        # prepare for adding eos
        sent_max_height = min(heights.max().item(), max_height)
        decoded: torch.LongTensor = current_best_beam.new(batch_size, self.num_beam_hyps_to_keep, sent_max_height, 4)
        # shorter batches are padded if needed
        if heights.min().item() != heights.max().item():
            decoded.fill_(-100)

        # fill with hypotheses and eos_token_id if the latter fits in
        for i, cur_best in enumerate(batch_best):
            for j, hypo in enumerate(cur_best):
                decoded[i, j, :heights[i,j]] = hypo
                decoded[i, j, heights[i, j] - 1][-1] = 1
        return UserDict(
            {
                "decoded": decoded,
                "best_scores": best_scores,
            }
        )


class BeamHypotheses:
    def __init__(self, num_beams: int, length_penalty: float):
        """
        Initialize n-best list of hypotheses.
        """
        # self.length_penalty = length_penalty
        self.num_beams = num_beams
        self.length_penalty = length_penalty
        self.beams = []
        self.worst_score = 1e9

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.beams)

    def add(self, hyp: torch.LongTensor, sum_logprobs: float):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / (hyp.shape[-2] ** self.length_penalty) ## height.
        if len(self) < self.num_beams or score > self.worst_score:
            self.beams.append((score, hyp))
            if len(self) > self.num_beams:
                sorted_next_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.beams)])
                del self.beams[sorted_next_scores[0][1]]
                self.worst_score = sorted_next_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs: float, cur_len: int) -> bool:
        """
        If there are enough hypotheses and that none of the hypotheses being generated can become better than the worst
        one in the heap, then we are done with this sentence.
        """

        if len(self) < self.num_beams:
            return False
        else:
            cur_score = best_sum_logprobs / (cur_len** self.length_penalty)
            ret = self.worst_score >= cur_score
            return ret
