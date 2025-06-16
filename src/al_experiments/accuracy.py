import string
import time
from typing import List, Tuple
import numpy as np
import pandas as pd
from statistics import mean, pstdev


class Accuracy:

    stored_accs = {}
    _letters = np.frombuffer(
        (string.ascii_lowercase + "AB").encode('ascii'),
        dtype=np.uint8
    )
    number_of_instances = 5355
    number_of_solvers = 28
    number_of_reduced_solvers = 27
    pairs = (number_of_reduced_solvers * (number_of_reduced_solvers - 1))
    instance_idx = np.arange(number_of_instances)

    def __init__(
            self,
            total_runtime,
            break_after_runtime_fraction,
            sample_result_after_iterations,
            sorted_rt: np.ndarray,
            par_2_scores,
            mean_par_2_score: float,
            par_2_score_removed_solver: float,
            runtime_of_removed_solver: np.ndarray
    ):
        self.total_runtime = total_runtime
        self.break_after_runtime_fraction = break_after_runtime_fraction
        self.sample_result_after_iterations = sample_result_after_iterations
        self.sorted_rt = sorted_rt
        self.par_2_scores = par_2_scores
        self.mean_par_2_score = mean_par_2_score
        self.par_2_score_removed_solver = par_2_score_removed_solver
        self.runtime_of_removed_solver = runtime_of_removed_solver
        self.n = 0
        self.used_runtime = 0
        self.pred = np.ascontiguousarray(
            np.full((27,), 0), dtype=np.float32
        )
        self.solver_results = []
        self.dtype = [('idx', np.int64), ('runtime', np.float64)]

    def add_runtime_quantized(
            self,
            thresholds: np.ndarray,
            prev_max_acc: float,
            prev_min_diff: float
    ):
        # instances not maxed out yet
        remaining_mask = thresholds < self.number_of_reduced_solvers
        valid_instances = self.instance_idx[remaining_mask]

        # current solver + its rt bearly solving the instance
        current_solver = self.sorted_rt[self.instance_idx, thresholds]

        # next solver + its rt that would solve the instance if threshold is raised
        next_solver = np.empty(self.number_of_instances, dtype=self.dtype)
        next_solver[:] = (-1, -1.)
        next_solver[valid_instances] = self.sorted_rt[valid_instances, thresholds[valid_instances] + 1]

        #print("extracted best next")
        #print(next_solver)

        # raising the thresh adds total_added_runtime seconds to instance i
        total_added_runtime = (
                self.number_of_reduced_solvers - thresholds
            ) * (next_solver['runtime'] - current_solver['runtime'])

        #print("total added runtime")
        #print(total_added_runtime)

        current_penalty = current_solver['runtime'] * 2
        #print("current_penalty")
        #print(current_penalty)
        next_penalty = next_solver['runtime'] * 2
        #print("next_penalty")
        #print(next_penalty)

        # copy previos pred to all instances
        new_pred = np.tile(self.pred, (self.number_of_instances, 1))
        # change pred for the next added solver
        next_solver['runtime'][next_solver['runtime'] == 5000] = 10000
        new_pred[self.instance_idx, next_solver['idx']] += (next_solver['runtime'] - current_penalty)

        # build a mask of which solvers still timeout with the new thresh
        index_mask = np.arange(self.number_of_solvers)[None, :] > thresholds[:, None] + 1
        index_mask = np.where(index_mask, self.sorted_rt['idx'] + 1, 0)
        timeout_mask = np.zeros_like(self.sorted_rt, dtype=bool)
        timeout_mask[self.instance_idx[:, None], index_mask] = True
        timeout_mask = timeout_mask[:, 1:]


        delta = next_penalty - current_penalty
        new_pred += timeout_mask * delta[:, None]
        #print("new pred with all instances")
        #print(new_pred)

        #print("mean pred")
        #print(new_pred.mean(axis=1))
        #print(new_pred.mean(axis=1).shape)


        scaling = self.mean_par_2_score / new_pred.mean(axis=1)
        #print("scaling")
        #print(scaling)
        #print(scaling.shape)

        similarity = np.abs((new_pred * scaling[:, None]) - self.par_2_scores).sum(axis=1)
        #print("similarity")
        #print(similarity)
        #print(similarity.shape)

        score = similarity #* total_added_runtime
        #print("fast")
        #for sc in score:
        #    print(sc, end=", ")

        #print()
        #print()
        #print("score")
        #print(score)
        #print(score.shape)
        #print(np.nanmin(score))

        best_idx = np.nanargmin(score[remaining_mask])
        best_idx = self.instance_idx[remaining_mask][best_idx]

        # update
        self.pred = new_pred[best_idx]
        thresholds[best_idx] += 1
        self.used_runtime += total_added_runtime[best_idx]
        self.n += 1

        if self.n % self.sample_result_after_iterations == 0:
            self.solver_results.append(
                self.sample_result(thresholds, score[best_idx])
            )
        if (self.used_runtime/self.total_runtime > self.break_after_runtime_fraction) or len(valid_instances) <= 1:
            return thresholds, prev_max_acc, -1
        return thresholds, prev_max_acc, prev_min_diff

    def add_runtime_random_quantized(
            self,
            thresholds: np.ndarray,
            prev_max_acc: float,
            prev_min_diff: float
    ):
        # instances not maxed out yet
        remaining_mask = thresholds < self.number_of_reduced_solvers
        valid_instances = self.instance_idx[remaining_mask]
        best_idx = np.random.choice(valid_instances)

        runtime_list = self.sorted_rt[best_idx]

        included_solvers = thresholds[best_idx]

        #print(f"currently, there are {included_solvers} solvers added")

        prev_solver, prev_added_runtime = runtime_list[included_solvers]

        #print(f"solver {prev_solver} is the current last. It has {prev_added_runtime}s runtime.")

        solver, added_runtime = runtime_list[included_solvers+1]

        #print(f"next solver is solver {solver}. It has {added_runtime}s runtime.")

        prev_penalty = prev_added_runtime * 2
        new_penalty = added_runtime * 2
        new_pred = self.pred.copy()
        new_pred[solver] += (added_runtime - prev_penalty)
        #print("here is the previous pred:")
        #print(self.pred)
        #print(f"the pred of solver {solver} is now {new_pred[solver]}")
        #print("here are the remaining preds:")
        new_arr = np.array(runtime_list, dtype=[('idx', np.int64), ('val', np.float64)])
        timeout_mask = new_arr['idx'][included_solvers+2:]
        new_pred[timeout_mask] += (new_penalty - prev_penalty)
        #print(new_pred)

        self.pred = new_pred
        thresholds[best_idx] += 1
        self.n += 1

        if self.n % self.sample_result_after_iterations == 0:
            self.solver_results.append(
                self.sample_result(thresholds)
            )
        if (self.used_runtime/self.total_runtime > self.break_after_runtime_fraction) or len(valid_instances) <= 1:
            return thresholds, prev_max_acc, -1
        return thresholds, prev_max_acc, prev_min_diff

    def sample_result(self, thresholds: np.ndarray, best_score=0):

        runtime_frac = self.used_runtime/self.total_runtime
        cross_acc = self.calc_cross_acc_2(self.par_2_scores, self.pred)

        new_pred = 0
        for index, runtime_list in enumerate(self.sorted_rt):
            _, timeout = runtime_list[thresholds[index]]
            # is instance maxed out?
            if thresholds[index] == self.number_of_reduced_solvers:
                # is solver runtime 5000?
                if self.runtime_of_removed_solver[index] == 5000:
                    new_pred += 10000
                else:
                    new_pred += self.runtime_of_removed_solver[index]
            elif timeout > self.runtime_of_removed_solver[index]:
                new_pred += self.runtime_of_removed_solver[index]
            else:
                new_pred += 2 * timeout
        all_par_2_scores = np.append(
            self.par_2_scores, self.par_2_score_removed_solver
        )
        all_pred = np.append(self.pred, new_pred)
        true_acc = self.calc_true_acc_1(
            all_par_2_scores,
            all_pred,
            self.number_of_reduced_solvers
        )
        print(f"actual key is {self.pred_vec_to_key(all_par_2_scores)}")
        print(f"pred key is   {self.pred_vec_to_key(all_pred)}")
        print(f"best score is {best_score}")
        print(f"cross acc is {cross_acc}")
        print(f"with this, the new total is {self.used_runtime} giving a fraction of {runtime_frac}")
        print(f"true acc is {true_acc}")
        return {
            "runtime_frac": runtime_frac,
            "cross_acc": cross_acc,
            "true_acc": true_acc,
            "diff": best_score
        }

    def sub_optimal_acc_maxing(self, new_acc: float, prev_acc: float):
        return new_acc <= prev_acc

    def sub_optimal_diff_mining(self, new_diff: float, prev_min: float):
        return new_diff >= prev_min

    def find_best_index_min_diff(
            self,
            thresholds: np.ndarray[np.floating[np.float32]],
            sorted_rt: np.ndarray,
            actu_par2,
            actu_par2_min: float,
            runtime_to_add: float,
            allowed_idxs:  np.ndarray = None,
            tol: float = 1e-8
    ):
        """
        Among only the positions in `allowed_idxs`, finds every index i
        for which `vec_to_diff(thresholds_with_x_at_i)` is minimal.

        Args:
        thresholds:  (M,)    array of original thresholds
        runtimes:    (M, N)  array of runtimes
        true_par_2:  (N,)    ground‑truth vector
        mean_actu:   scalar  precomputed true_par_2.mean()
        x:            scalar  the increment to test at each threshold
        allowed_idxs:(K,)    integer indices into [0..M)
        tol:         scalar  tolerance for floating‑point ties

        Returns:
        best_idxs:   1D array of original indices (subset of allowed_idxs)
        best_val:    the minimal similarity value
        """
        # -- restrict to allowed subset ---------------------------------------
        # extract only the K rows we care about
        if allowed_idxs is not None:
            thresholds = thresholds[allowed_idxs]          # (5355,)
            runtimes = runtimes[allowed_idxs, :]         # (5355, 27)



        new_alive_instances = np.eye(self.number_of_instances, dtype=bool)
        already_alive_instances = thresholds != 0
        take_old_instances = already_alive_instances * ~new_alive_instances

        number_of_living_instances = already_alive_instances.sum()
        one_more_living_instance = number_of_living_instances + 1


        # 1) original score‐matrix on the subset
        old_scores = self.replace_by_overflow_mean(runtimes, thresholds)             # (5355, 27)

        #print("old scores")
        #print(old_scores)

        old_par_2 = old_scores.mean(axis=0)               # (27,)

        #print("old scores")
        #print(old_scores)

        # 2) scores with thresholds + x on the subset
        new_thresh = thresholds + runtime_to_add
        valid_mask = (new_thresh < 5000)
        new_scores_adding_thresh_to_every_instance = self.replace_by_overflow_mean(runtimes, new_thresh)        # (5355, 27)

        total_solver_runtime = old_scores[already_alive_instances, :].sum(axis=0)  # (27,)

        new_total_runtime_adding_thresh_to_instance_i = np.where(already_alive_instances[:, None], total_solver_runtime[None, :] + ((new_scores_adding_thresh_to_every_instance - old_scores)), total_solver_runtime[None, :] + (new_scores_adding_thresh_to_every_instance))


        #print(pd.DataFrame(new_total_runtime_adding_thresh_to_instance_i))
        # 4) compute similarity for each candidate
        total_min = new_total_runtime_adding_thresh_to_instance_i.min(axis=1)    # (5355,)
        temp = new_total_runtime_adding_thresh_to_instance_i - total_min[:, None]
        means = np.mean(temp, 1)
        means[means == 0] = 0.001
        error_per_solver_per_selected_instance = np.abs((temp) * (304.823517/means)[:, None] - (actu_par2[None, :] - actu_par2_min))  # (5355, 27)

        total_error_per_selected_instance = error_per_solver_per_selected_instance.sum(axis=1)      # (5355,)
        best_total_error = total_error_per_selected_instance[valid_mask].min()
        print(f"best={best_total_error}")

        # 6) pick all within tol of the minimum
        best_mask = np.isclose(total_error_per_selected_instance, best_total_error, atol=tol)
        if allowed_idxs is None:
            best_idxs = np.where(best_mask & valid_mask)[0]
        else:
            best_idxs = allowed_idxs[best_mask & valid_mask]

        return best_idxs, best_total_error

    def find_all_best_indices_max_cross_acc(
            self,
            thresholds: np.ndarray[np.floating[np.float32]],
            runtimes: np.ndarray[np.floating[np.float32]],
            actu_par_2,
            mean_actu: float,
            runtime_to_add: float,
            allowed_idxs:  np.ndarray = None,
            punishment: float = 10000.0,
            tol: float = 1e-8
    ):
        """
        Among all i=0..M-1, returns every index i for which
        vec_to_cross_acc(thresholds with thresholds[i]+=x)
        is maximal (within `tol`), along with that max accuracy.

        vec_to_cross_acc: uses vec_to_pred(..., punishment) and
        calc_cross_acc_2, i.e. the fraction of concordant ordered pairs.

        NOTE: this builds an (M×N×N) boolean tensor, so memory is O(M*N^2).
        """

        # -- restrict to allowed subset ---------------------------------------
        # extract only the K rows we care about
        if allowed_idxs is not None:
            #print("allowed_idxs:")
            #print(allowed_idxs)
            thresholds = thresholds[allowed_idxs]          # (5355,)
            runtimes = runtimes[allowed_idxs, :]         # (5355, allowed_idxs.size)

        # 1) base score‐matrix & prediction
        thresholds = np.ascontiguousarray(thresholds, dtype=np.float32)
        runtimes = np.ascontiguousarray(runtimes, dtype=np.float32)

        old_scores = self.replace_by_overflow_mean(runtimes, thresholds)    # (M, N)
        old_par_2 = old_scores.mean(axis=0)                # (N,)
        #print("old scores:")
        #print(old_scores)

        #print("old par_2_scores:")
        #print(old_par_2)

        # 2) score‐matrix with thresholds + x
        new_thresh = thresholds + runtime_to_add
        valid_mask = new_thresh < 5000
        new_scores_adding_thresh_to_every_instance = self.replace_by_overflow_mean(runtimes, new_thresh)          # (5355, allowed_idxs.size)

        #print(f"new scores when adding {runtime_to_add} to thresh:")
        #print(new_scores_adding_thresh_to_every_instance)

        # 3) candidate predictions for each i
        # (new_scores - old_scores) / M is change par_2_score
        new_par_2_scores_when_adding_thresh_to_instance_i = old_par_2[None, :] + (new_scores_adding_thresh_to_every_instance - old_scores) / self.number_of_instances   # (5355, allowed_idxs.size)

        #print("new par_2")

        #print(new_par_2_scores_when_adding_thresh_to_instance_i)

        # 4) compute cross‐accuracy for each i
        # dp: for each candidate i, and each ordered pair of samples (j,k)(j,k), dp[i,j,k] is the difference in predicted scores between sample jj and sample kk.;
        # da: true differences
        dp = new_par_2_scores_when_adding_thresh_to_instance_i[:, :, None] - new_par_2_scores_when_adding_thresh_to_instance_i[:, None, :]       # (5355, allowed_idxs.size, allowed_idxs.size)
        da = actu_par_2[:, None] - actu_par_2[None, :]   # (N, N)

        # concordant if dp * da > 0
        concordant = (dp * da[None, :, :]) > 0          # (5355, allowed_idxs.size, allowed_idxs.size)
        concordant_count = concordant.sum(axis=(1, 2))   # (5355,)

        # average over n*(n-1) ordered pairs
        accs = concordant_count / self.pairs        # (5355,)
        best_acc = accs[valid_mask].max()

        # 6) pick all indices within tol of the max
        best_mask = np.isclose(accs, best_acc, atol=tol)
        if allowed_idxs is None:
            best_idx = np.where(best_mask & valid_mask)[0]
        else:
            best_idx = allowed_idxs[best_mask & valid_mask]

        return best_idx, best_acc

    def is_new_sum_correct(self, sum: np.ndarray, new_scores_adding_thresh_to_every_instance, thresholds, old_scores):
        par_2_choosing_instance_i = []
        for i in range(5355):
            total_runtime = []
            for j in range(5355):
                if i == j:
                    total_runtime.append(new_scores_adding_thresh_to_every_instance[j])
                elif thresholds[j] == 0:
                    continue
                else:
                    total_runtime.append(old_scores[j])
            all_runtimes = pd.DataFrame(total_runtime)
            par_2_choosing_instance_i.append(all_runtimes.sum(axis=0))
        correct_sums = pd.DataFrame(par_2_choosing_instance_i)
        print("manual")
        print(correct_sums)
        print("faster")
        print(pd.DataFrame(sum))

    def replace_by_overflow_mean(self, values: np.ndarray, limits: np.ndarray):
        """
        values: (M, N)
        limits: (M,)
        returns: (M, N) where each row[i,j] > limits[i] is
                 replaced by mean(values[i, mask_i])
        """
        mask = values > limits[:, None]           # (M, N)
        sums = np.sum(values * mask, axis=1)    # (M,)
        counts = np.sum(mask, axis=1)        # (M,)
        with np.errstate(divide='ignore', invalid='ignore'):
            overflow_mean = sums / counts         # (M,)
        # do the replacement
        return np.where(mask, overflow_mean[:, None], values)

    def vec_to_pred(
            self,
            thresholds: np.ndarray[np.floating[np.float32]],
            runtimes: np.ndarray[np.floating[np.float32]],
            punishment: int = 10000
    ):
        """
        thresholds: 1D array of shape (M,)
        runtimes:    2D array of shape (M, N)

        For each i:
        let mask_i = runtimes[i, :] > thresholds[i]
        compute mean_i = mean(runtimes[i, mask_i])
        replace each runtimes[i, j] where mask_i[j] is True with mean_i
        then return the column‐wise mean of the resulting array.
        """
        # ensure flat, contiguous float32 arrays
        thr = np.ascontiguousarray(thresholds, dtype=np.float32).ravel()
        runs = np.ascontiguousarray(runtimes,   dtype=np.float32)

        # build boolean mask of “over‐threshold” entries
        mask = runs > thr[:, None]      # shape (M, N)

        # sum and count of over‐threshold per row
        # runs * mask casts mask to 0/1, so sums only the True elements
        sums = np.sum(runs * mask, axis=1)    # shape (M,)
        counts = np.sum(mask,      axis=1)      # shape (M,)

        # row‐means of the values > threshold; 
        # we suppress divide‐by‐zero warnings since if counts[i]==0,
        # that row has no masked entries and mean[i] will not actually be used.
        with np.errstate(divide='ignore', invalid='ignore'):
            mean_over = sums / counts             # shape (M,)

        # replace over‐threshold entries by the corresponding row mean
        # broadcasting mean_over[:, None] to shape (M, N)
        replaced = np.where(mask, mean_over[:, None], runs)

        # finally, return the column‐wise mean
        return replaced.mean(axis=0)

    def vec_to_pred_punish_thresh(
            self,
            thresholds: np.ndarray[np.floating[np.float32]],
            runtimes: np.ndarray[np.floating[np.float32]],
    ):
        """
        thresholds: 1D array‑like of shape (5355,)
        runtimes:    2D array‑like of shape (5355, N)

        For each i, any runtimes[i, j] > thresholds[i] is replaced
        by punishment, then averaged across i
        """
        thresholds = np.ascontiguousarray(thresholds, dtype=np.float32)
        runtimes = np.ascontiguousarray(runtimes,  dtype=np.float32)

        # broadcast compare & clamp
        # thr[:, None] gives shape (N,1) so thr[i] is compared to run[i,j]
        scores = np.where(
            runtimes > thresholds[:, None], 2*thresholds[:, None], runtimes
        )

        return scores.mean(axis=0)

    def print_key_signature(
            self,
            thresholds: np.ndarray[np.floating[np.float32]],
            runtimes: np.ndarray[np.floating[np.float32]],
            actu
    ):
        pred = self.vec_to_pred(thresholds, runtimes)
        print(f"actual key is {self.pred_vec_to_key(actu)}")
        print(f"pred key is   {self.pred_vec_to_key(pred)}")

    def vec_to_diff(
            self,
            thresholds: np.ndarray[np.floating[np.float32]],
            runtimes: np.ndarray[np.floating[np.float32]],
            true_par_2,
            true_par_2_mean: float,
            removed_index: int
    ):
        pred_par_2 = np.delete(self.vec_to_pred(thresholds, runtimes), removed_index)

        return self.similarity(true_par_2, pred_par_2, true_par_2_mean)

    def vec_to_cross_acc(
            self,
            thresholds: np.ndarray[np.floating[np.float32]],
            runtimes: np.ndarray[np.floating[np.float32]],
            true_par_2
    ):
        pred_par_2 = self.vec_to_pred(thresholds, runtimes, 10000)

        acc = self.calc_cross_acc_2(true_par_2, pred_par_2)

        return acc

    def vec_to_true_acc(
            self,
            thresholds: np.ndarray[np.floating[np.float32]],
            runtimes: np.ndarray[np.floating[np.float32]],
            true_par_2: np.ndarray,
            index: int
    ):
        pred_par_2_for_this_solver = self.vec_to_pred(thresholds, runtimes, 10000)[index]

        pred_par_2 = true_par_2.copy()
        pred_par_2[index] = pred_par_2_for_this_solver
        return self.calc_true_acc_1(true_par_2, pred_par_2, index)

    def vec_to_true_acc_2(
            self,
            thresholds: np.ndarray[np.floating[np.float32]],
            runtimes: np.ndarray[np.floating[np.float32]],
            true_par_2: np.ndarray,
            index: int      
    ):
        scores = self.replace_by_overflow_mean(runtimes, thresholds)

    def determine_acc(self, actu: np.ndarray, pred: np.ndarray):
        key = self.pred_vec_to_key(pred)

        # no uniue ordering found
        if key is None:
            # return without storing
            return self.calc_cross_acc_2(actu, pred)

        # key is stored
        if key in self.stored_accs:
            # return stored value
            return self.stored_accs[key]

        # else calculate, store and return value
        acc = self.calc_cross_acc_2(actu, pred)
        self.stored_accs[key] = acc
        return acc

    def pred_vec_to_key(self, pred: np.ndarray):
        """
        Sorts a 1D C‑contiguous float array `pred` (len ≤ 28) and returns
        a string of letters corresponding to the original indices in ascending order.
        returns None if any values in `pred` are exactly duplicated.
        """
        # 1) get sorted indices & sorted values
        idx = np.argsort(pred, kind='quicksort')
        sv = pred[idx]

        # 2) detect duplicates in one vectorized call
        if np.any(sv[1:] == sv[:-1]):
            return None

        # 3) map sorted original‐indices → ASCII codes → bytes → string
        letters_u8 = self._letters[idx]
        return bytes(letters_u8).decode('ascii')

    def similarity(self, actu: np.ndarray, pred: np.ndarray, mean_actu: float):
        """
        L1‐distance after scaling pred so its mean matches mean_actu
        """
        scaling = mean_actu / pred.mean()
        return np.abs((pred * scaling) - actu).sum()

    def calc_cross_acc_1(
            self,
            par_2_scores: pd.Series,
            predicted_par_2_scores: pd.Series
    ):

        result_df = pd.concat([par_2_scores, predicted_par_2_scores], axis=1)
        result_df['rank_accuracy'] = np.nan

        for index_1, value_1 in predicted_par_2_scores.items():
            rank_accuracy = 0
            for index_2, value_2 in predicted_par_2_scores.items():
                if (value_2 - value_1) * (par_2_scores[index_2] - par_2_scores[index_1]) > 0 or index_1 == index_2:
                    rank_accuracy += 1/par_2_scores.size
            result_df.at[index_1, 'rank_accuracy'] = rank_accuracy

        average = result_df['rank_accuracy'].mean(skipna=True)

        return average

    def calc_cross_acc_2(
            self, actu: np.ndarray, pred: np.ndarray
    ) -> float:
        # compute all pairwise differences
        # shape is (n, n)
        dp = pred[:, None] - pred[None, :]
        da = actu[:, None] - actu[None, :]

        # a concordant (correct) pair is where dp and da have the same sign
        # (= dp * da > 0)
        concordant = np.count_nonzero(dp * da > 0)

        # total possible ordered comparisons per solver is (n - 1),
        # and we average over n solvers:
        #   average = (1/(n*(n-1))) * concordant
        return concordant / (actu.size * (actu.size - 1))

    def calc_cross_acc_3(self, actu: np.ndarray, pred: np.ndarray):
        # Compute pairwise differences
        pred_diff = pred[:, None] - pred
        true_diff = actu[:, None] - actu

        # Check for agreement in direction
        agreement = (pred_diff * true_diff) > 0

        # Sum up agreements per row and normalize
        rank_accuracies = agreement.sum(axis=1) / (actu.size - 1)

        return rank_accuracies.mean()

    def calc_true_acc_1(self, actu: np.ndarray, pred: np.ndarray, index: int):
        acc = 0
        pred_index = pred[index]
        actu_index = actu[index]
        solvers = actu.size
        for i in range(solvers):
            if (pred[i] - pred_index) * (actu[i] - actu_index) > 0:
                acc += 1
        return acc / (solvers - 1)

    def calc_true_acc_2(self, actu: np.ndarray, pred: np.ndarray, index: int) -> float:
        """
        DEPRECATED! implemented with A and not A'
        """
        # Compute signed differences for given reference
        dp = pred - pred[index]
        da = actu - actu[index]
        # Element‐wise product > 0 gives a boolean array of correct‐sign matches
        correct = (dp * da) > 0
        # Mean of booleans is the fraction of True values
        return np.mean(correct)
