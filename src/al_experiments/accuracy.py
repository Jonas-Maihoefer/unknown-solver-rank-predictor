import string
import numpy as np
import pandas as pd


class accuracy:

    stored_accs = {}
    _letters = np.frombuffer((string.ascii_lowercase + "AB").encode('ascii'), dtype=np.uint8)

    def vector_to_cross_acc(
            self,
            thresholds: np.ndarray[np.floating[np.float32]],
            runtimes: np.ndarray[np.floating[np.float32]],
            true_par_2
    ):
        """
        thresholds: 1D array‑like of shape (5355,)
        runtimes:    2D array‑like of shape (5355, 28)

        For each i, any runtimes[i, j] > thresholds[i] is replaced by 10000,
        then averaged across i
        """
        thresholds = np.ascontiguousarray(thresholds, dtype=np.float32)
        runtimes = np.ascontiguousarray(runtimes,  dtype=np.float32)

        # broadcast compare & clamp
        # thr[:, None] gives shape (N,1) so thr[i] is compared to run[i,j]
        scores = np.where(runtimes > thresholds[:, None], 10000.0, runtimes)

        pred_par_2 = scores.mean(axis=0)

        acc = self.calc_cross_acc_2(true_par_2, pred_par_2)

        return acc

    def vector_to_true_acc(
            self,
            thresholds: np.ndarray[np.floating[np.float32]],
            runtimes: np.ndarray[np.floating[np.float32]],
            true_par_2,
            index: int
    ):
        """
        thresholds: 1D array‑like of shape (5355,)
        runtimes:    2D array‑like of shape (5355, 28)

        For each i, any runtimes[i, j] > thresholds[i] is replaced by 10000,
        then averaged across i
        """
        thresholds = np.ascontiguousarray(thresholds, dtype=np.float32)
        runtimes = np.ascontiguousarray(runtimes,  dtype=np.float32)

        # broadcast compare & clamp
        # thr[:, None] gives shape (N,1) so thr[i] is compared to run[i,j]
        scores = np.where(runtimes > thresholds[:, None], 10000.0, runtimes)

        pred_par_2 = scores.mean(axis=0)

        acc = self.calc_true_acc(true_par_2, pred_par_2, index)

        print(f"actual key is {self.pred_vec_to_key(true_par_2)}")

        print(f"pred key is   {self.pred_vec_to_key(pred_par_2)}")  

        return acc

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

    def calc_true_acc(self, actu: np.ndarray, pred: np.ndarray, index: int):
        acc = 0
        pred_index = pred[index]
        actu_index = actu[index]
        solvers = actu.size
        for i in range(solvers):
            if (pred[i] - pred_index) * (actu[i] - actu_index) > 0:
                acc += 1
        return acc / solvers
