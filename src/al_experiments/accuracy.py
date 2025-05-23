import string
import time
import numpy as np
import pandas as pd
from statistics import mean, pstdev


class accuracy:

    stored_accs = {}
    _letters = np.frombuffer((string.ascii_lowercase + "AB").encode('ascii'), dtype=np.uint8)
    number_of_instances = 5355
    number_of_reduced_solvers = 27
    pairs = (number_of_reduced_solvers * (number_of_reduced_solvers - 1))
    n = 0

    def add_runtime(
            self,
            thresholds: np.ndarray[np.floating[np.float32]],
            runtimes: np.ndarray[np.floating[np.float32]],
            par_2_scores,
            mean_par_2_score: float,
            runtime_to_add: float,
            prev_max_acc: float,
            prev_min_diff: float
    ):
        """ while (True):
            if self.n % 2 == 0:
                best_instances, max_acc = self.find_all_best_indices_max_cross_acc(thresholds, runtimes, par_2_scores, mean_par_2_score, runtime_to_add)
                if (best_instances.size == 0):
                    return thresholds, prev_max_acc, -1 
                best_instances, min_diff = self.find_best_index_min_diff(thresholds, runtimes, par_2_scores, mean_par_2_score, runtime_to_add, best_instances)
                if self.sub_optimal_acc_maxing(max_acc, prev_max_acc):
                    runtime_to_add *= 2
                    if runtime_to_add >= 5000:
                        self.n += 1
                        return thresholds, prev_max_acc * 0.99, prev_min_diff
                    print(f"again with {runtime_to_add}")
                else:
                    break
            else:
                best_instances, min_diff = self.find_best_index_min_diff(thresholds, runtimes, par_2_scores, mean_par_2_score, runtime_to_add)
                if (best_instances.size == 0):
                    return thresholds, prev_max_acc, -1 
                best_instances, max_acc = self.find_all_best_indices_max_cross_acc(thresholds, runtimes, par_2_scores, mean_par_2_score, runtime_to_add, best_instances)
                if self.sub_optimal_diff_mining(min_diff, prev_min_diff):
                    runtime_to_add *= 2
                    if runtime_to_add >= 5000:
                        self.n += 1
                        return thresholds, prev_max_acc, prev_min_diff * 1.01
                    print(f"again with {runtime_to_add}")
                else:
                    break """
        
        random_index = np.random.randint(0, thresholds.size)
        while thresholds[random_index] + runtime_to_add >= 5000:
            random_index = np.random.randint(0, thresholds.size)

        thresholds[random_index] += runtime_to_add
        """self.n += 1"""
        return thresholds, 0, 0

    def sub_optimal_acc_maxing(self, new_acc: float, prev_acc: float):
        return new_acc <= prev_acc

    def sub_optimal_diff_mining(self, new_diff: float, prev_min: float):
        return new_diff >= prev_min

    def find_best_index_min_diff(
            self,
            thresholds: np.ndarray[np.floating[np.float32]],
            runtimes: np.ndarray[np.floating[np.float32]],
            actu,
            mean_actu: float,
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
            runtimes = runtimes[allowed_idxs, :]         # (5355, allowed_idxs.size)


        # 1) original score‐matrix on the subset
        old_scores = np.where(
            runtimes > thresholds[:, None],
            2 * thresholds[:, None],
            runtimes
        )             # (5355, allowed_idxs.size)

        #print("old scores")
        #print(old_scores)

        old_par_2 = old_scores.mean(axis=0)               # (allowed_idxs.size,)

        #print("old scores")
        #print(old_scores)

        # 2) scores with thresholds + x on the subset
        new_thresh = thresholds + runtime_to_add
        valid_mask = (new_thresh < 5000)
        new_scores_adding_thresh_to_every_instance = np.where(
            runtimes > new_thresh[:, None],
            2 * new_thresh[:, None],
            runtimes
        )            # (5355, allowed_idxs.size)

        # 3) candidate predictions for each allowed i
        new_par_2_scores_when_adding_thresh_to_instance_i = old_par_2[None, :] + (new_scores_adding_thresh_to_every_instance - old_scores) / self.number_of_instances  # (5355, allowed_idxs.size)

        # 4) compute similarity for each candidate
        preds_mean_sub = new_par_2_scores_when_adding_thresh_to_instance_i.mean(axis=1)    # (5355,)
        scalings_sub = mean_actu / preds_mean_sub
        errs_sub = np.abs(new_par_2_scores_when_adding_thresh_to_instance_i * scalings_sub[:, None] - actu[None, :])
        sims_sub = errs_sub.sum(axis=1)      # (5355,)
        best_val = sims_sub[valid_mask].min()

        # 6) pick all within tol of the minimum
        best_mask = np.isclose(sims_sub, best_val, atol=tol)
        if allowed_idxs is None:
            best_idxs = np.where(best_mask & valid_mask)[0]
        else:
            best_idxs = allowed_idxs[best_mask & valid_mask]

        return best_idxs, best_val

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
        old_scores = np.where(
            runtimes > thresholds[:, None],
            punishment,
            runtimes
        )          # (5355, allowed_idxs.size)
        #print("old scores:")
        #print(old_scores)

        old_par_2 = old_scores.mean(axis=0)                   # (allowed_idxs.size,)

        #print("old par_2_scores:")
        #print(old_par_2)

        # 2) score‐matrix with thresholds + x
        new_thresh = thresholds + runtime_to_add
        valid_mask = new_thresh < 5000
        new_scores_adding_thresh_to_every_instance = np.where(
            runtimes > new_thresh[:, None],
            punishment,
            runtimes
        )          # (5355, allowed_idxs.size)

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
            true_par_2_mean: float
    ):
        pred_par_2 = self.vec_to_pred_punish_thresh(thresholds, runtimes)

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
            true_par_2,
            index: int
    ):
        pred_par_2 = self.vec_to_pred(thresholds, runtimes, 10000)

        return self.calc_true_acc_1(true_par_2, pred_par_2, index)

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

    def pred_vec_to_key(self, pred: np.ndarray) -> int | None:
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
