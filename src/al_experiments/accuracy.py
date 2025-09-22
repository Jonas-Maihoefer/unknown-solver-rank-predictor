import string
import pandas as pd
import os
from al_experiments.constants import Constants, idx, rt

# number_of_solvers, number_of_reduced_solvers, reduced_solver_pairs, number_of_instances, instance_idx,

useCupy = os.getenv("USECUDA", "0") == "1"

if useCupy:
    import cupy as np
else:
    import numpy as np


class Accuracy:

    stored_accs = {}
    _letters = np.frombuffer(
        (string.ascii_lowercase + "AB").encode('ascii'),
        dtype=np.uint8
    )

    def __init__(
            self,
            con: Constants,
            total_runtime,
            total_rt_removed_solver,
            break_after_runtime_fraction,
            sample_result_after_iterations,
            sorted_rt,
            sorted_rt_rated,
            par_2_scores,
            mean_par_2_score: float,
            par_2_score_removed_solver: float,
            runtime_of_removed_solver,
            select_idx,
            solver_string,
            all_results,
            scoring_fn,
            thresh_breaking_condition,
            rt_weight: float = 0.0,
            with_remaining_mean: bool = False
    ):
        self.number_of_solvers = con.number_of_solvers
        self.number_of_reduced_solvers = con.number_of_reduced_solvers
        self.reduced_solver_pairs = con.reduced_solver_pairs
        self.number_of_instances = con.number_of_instances
        self.instance_idx = con.instance_idx
        self.total_runtime = total_runtime
        self.break_after_runtime_fraction = break_after_runtime_fraction
        self.sample_result_after_iterations = sample_result_after_iterations
        print(f"sampling result after {sample_result_after_iterations} interations.")
        self.sorted_rt = sorted_rt
        self.sorted_rt_rated = sorted_rt_rated
        self.par_2_scores = par_2_scores
        self.mean_par_2_score = mean_par_2_score
        self.par_2_score_removed_solver = par_2_score_removed_solver
        self.rt_removed_solver = runtime_of_removed_solver
        self.select_idx = select_idx
        self.solver_string = solver_string
        self.all_results = all_results
        self.rt_weight = rt_weight
        self.with_remaining_mean = with_remaining_mean
        self.total_rt_removed_solver = total_rt_removed_solver
        self.scoring_fn = scoring_fn
        self.thresh_breaking_condition = thresh_breaking_condition
        self.n = 1
        self.used_runtime = 0
        self.pred = np.ascontiguousarray(
            np.full((self.number_of_reduced_solvers,), 0), dtype=np.float32
        )

    def add_runtime_quantized(
            self,
            thresholds,
            prev_max_acc: float,
            prev_min_diff: float
    ):
        # instances not maxed out yet
        remaining_mask = thresholds < self.number_of_reduced_solvers
        valid_instances = self.instance_idx[remaining_mask]

        # current solver + its rt bearly solving the instance
        current_solver = self.sorted_rt[idx][self.instance_idx, thresholds], self.sorted_rt[rt][self.instance_idx, thresholds]

        # next solver + its rt that would solve the instance if threshold is raised
        next_solver = (
            np.empty(self.number_of_instances, dtype=int),
            np.empty(self.number_of_instances, dtype=float)
        )
        next_solver[idx][:] = -1
        next_solver[rt][:] = -1
        next_solver[idx][valid_instances] = self.sorted_rt[idx][valid_instances, thresholds[valid_instances] + 1]
        next_solver[rt][valid_instances] = self.sorted_rt[rt][valid_instances, thresholds[valid_instances] + 1]

        #print("extracted best next")
        #print(next_solver)

        # raising the thresh adds total_added_runtime seconds to instance i
        total_added_runtime = (
                self.number_of_reduced_solvers - thresholds
            ) * (next_solver[rt] - current_solver[rt])

        #print("total added runtime")
        #print(total_added_runtime)

        current_penalty = current_solver[rt] * 2
        #print("current_penalty")
        #print(current_penalty)
        next_penalty = next_solver[rt] * 2
        #print("next_penalty")
        #print(next_penalty)

        # copy previos pred to all instances
        new_pred = np.tile(self.pred, (self.number_of_instances, 1))
        # change pred for the next added solver
        next_solver[rt][next_solver[rt] == 5000] = 10000

        new_pred[self.instance_idx, next_solver[idx]] += (next_solver[rt] - current_penalty)

        # build a mask of which solvers still timeout with the new thresh
        index_mask = np.arange(self.number_of_solvers)[None, :] > thresholds[:, None] + 1
        index_mask = np.where(index_mask, self.sorted_rt[idx] + 1, 0)
        timeout_mask = np.zeros_like(self.sorted_rt[idx], dtype=bool)
        timeout_mask[self.instance_idx[:, None], index_mask] = True
        timeout_mask = timeout_mask[:, 1:]

        delta = next_penalty - current_penalty
        new_pred += timeout_mask * delta[:, None]
        #print("new pred with all instances")
        #print(new_pred)

        #print("mean pred")
        #print(new_pred.mean(axis=1))
        #print(new_pred.mean(axis=1).shape)

        #print("similarity")
        #print(similarity[remaining_mask])

        #print("similarity")
        #print(similarity)
        #print(similarity.shape)

        #print("total_added_runtime")
        #print(total_added_runtime[remaining_mask])

        score = self.scoring_fn(new_pred, self.par_2_scores, total_added_runtime, self.rt_weight)

        #print("score")
        #print(score[remaining_mask])

        # TODO: remove!
        bad_indices = np.where(np.isneginf(score[remaining_mask]) | np.isnan(score[remaining_mask]))[0]
        if bad_indices.size > 0:
            for bad_idx in bad_indices:
                bad_idx = self.instance_idx[remaining_mask][bad_idx]
                print(f"135000000 / ({score[bad_idx]} * {total_added_runtime[bad_idx]}) = {135000000 / (similarity[bad_idx] * total_added_runtime[bad_idx])}")
                print(new_pred[bad_idx])
                print(f"total_added_runtime is ({self.number_of_reduced_solvers} - {thresholds[bad_idx]}) * ({next_solver[rt][bad_idx]} - {current_solver[rt][bad_idx]}) = {total_added_runtime[bad_idx]}")

        #print("fast")
        #for th in thresholds:
        #    print(th, end=", ")

        #print()
        #print()
        #print("score")
        #print(score)
        #print(score.shape)
        #print(np.nanmin(score))

        if not remaining_mask.any():
            print("No more thresholds remaining")
            return thresholds, prev_max_acc, -1

        best_idx = self.select_idx(score, remaining_mask, self.instance_idx)

        # update
        self.pred = new_pred[best_idx]
        thresholds[best_idx] += 1
        self.used_runtime += total_added_runtime[best_idx]

        if self.n % self.sample_result_after_iterations == 0:
            runtime_frac, cross_acc, stability = self.sample_result(
                thresholds, self.pred,
                "determine_timeouts", score[best_idx]
            )
            if runtime_frac > self.break_after_runtime_fraction:
                return thresholds, prev_max_acc, -1
            if self.thresh_breaking_condition.fn(runtime_frac, cross_acc, stability):
                return thresholds, prev_max_acc, -1
        self.n += 1
        return thresholds, prev_max_acc, prev_min_diff

    def add_runtime_quantized_mean_punish(
            self,
            thresholds,
            prev_max_acc: float,
            prev_min_diff: float
    ):
        # instances not maxed out yet
        remaining_mask = thresholds < self.number_of_reduced_solvers
        valid_instances = self.instance_idx[remaining_mask]

        # current solver + its rt bearly solving the instance
        current_solver = self.sorted_rt[idx][self.instance_idx, thresholds], self.sorted_rt[rt][self.instance_idx, thresholds]

        # next solver + its rt that would solve the instance if threshold is raised
        next_solver = (
            np.empty(self.number_of_instances, dtype=int),
            np.empty(self.number_of_instances, dtype=float)
        )
        next_solver[idx][:] = -1
        next_solver[rt][:] = -1
        next_solver[idx][valid_instances] = self.sorted_rt[idx][valid_instances, thresholds[valid_instances] + 1]
        next_solver[rt][valid_instances] = self.sorted_rt[rt][valid_instances, thresholds[valid_instances] + 1]

        #print("extracted best next")
        #print(next_solver)

        # raising the thresh adds total_added_runtime seconds to instance i
        total_added_runtime = (
                self.number_of_reduced_solvers - thresholds
            ) * (next_solver[rt] - current_solver[rt])

        #print("total added runtime")
        #print(total_added_runtime)

        current_penalty = self.get_remaining_mean(thresholds)
        #print("current_penalty")
        #print(current_penalty)
        next_penalty = self.get_remaining_mean(thresholds, offset=1)
        #print("next_penalty")
        #print(next_penalty)

        # copy previos pred to all instances
        new_pred = np.tile(self.pred, (self.number_of_instances, 1))
        # change pred for the next added solver
        next_solver[rt][next_solver[rt] == 5000] = 10000

        new_pred[self.instance_idx, next_solver[idx]] += (next_solver[rt] - current_penalty) / self.number_of_instances

        # build a mask of which solvers still timeout with the new thresh
        index_mask = np.arange(self.number_of_solvers)[None, :] > thresholds[:, None] + 1
        index_mask = np.where(index_mask, self.sorted_rt[idx] + 1, 0)
        timeout_mask = np.zeros_like(self.sorted_rt[idx], dtype=bool)
        timeout_mask[self.instance_idx[:, None], index_mask] = True
        timeout_mask = timeout_mask[:, 1:]

        delta = next_penalty - current_penalty
        new_pred += (timeout_mask * delta[:, None])
        #print("new pred with all instances")
        #print(new_pred)

        #print("mean pred")
        #print(new_pred.mean(axis=1))
        #print(new_pred.mean(axis=1).shape)

        # TODO: the next 3 lines get outsourced linke in _double_punish

        similarity = batch_rmse(new_pred, self.par_2_scores)
        #print("similarity")
        #print(similarity)
        #print(similarity.shape)

        score = 135000000 / np.float_power(similarity, self.rt_weight)

        profitability_index = score / total_added_runtime

        #print("fast")
        #for th in thresholds:
        #    print(th, end=", ")

        #print()
        #print()
        #print("score")
        #print(score)
        #print(score.shape)
        #print(np.nanmin(score))

        if not remaining_mask.any():
            print("No more thresholds remaining")
            return thresholds, prev_max_acc, -1

        best_idx = self.select_idx(profitability_index, remaining_mask, self.instance_idx)

        # update
        self.pred = new_pred[best_idx]
        #print("choosen pred")
        #print(self.pred)
        thresholds[best_idx] += 1
        self.used_runtime += total_added_runtime[best_idx]

        if self.n % self.sample_result_after_iterations == 0:
            runtime_frac, cross_acc, stability = self.sample_result(
                thresholds, self.pred,
                "determine_timeouts", profitability_index[best_idx]
            )
            if runtime_frac > self.break_after_runtime_fraction:
                return thresholds, prev_max_acc, -1
            if runtime_frac > 0.15 and cross_acc >= 1.0:
                return thresholds, prev_max_acc, -1
        self.n += 1
        return thresholds, prev_max_acc, prev_min_diff

    def get_remaining_mean(self, thresholds, offset=0):
        # print(self.sorted_rt[rt])
        col_idx = np.arange(self.number_of_solvers)
        remaining_instances = col_idx >= thresholds[:, None] + (offset + 1)
        sums = (self.sorted_rt_rated[rt] * remaining_instances).sum(axis=1)
        counts = remaining_instances.sum(axis=1)

        # avoid division-by-zero if any threshold==m
        averages = sums / np.where(counts == 0, 1, counts)
        averages = np.where(counts == 0, 10000, averages)
        return averages

    def add_runtime_random_quantized(
            self,
            thresholds,
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

        if self.n % self.sample_result_after_iterations == 0:
            runtime_frac, cross_acc, stability = self.sample_result(
                thresholds, self.pred, "determine_timeouts"
            )
            if runtime_frac > self.break_after_runtime_fraction:
                return thresholds, prev_max_acc, -1
        self.n += 1
        return thresholds, prev_max_acc, prev_min_diff

    def linear_fit(self, x: np.ndarray, y: np.ndarray):
        """
        Given two 1D arrays x and y of the same length,
        finds m, c minimizing ||y - (m*x + c)||_2 and
        returns (m, c, e), where e is the RMSE of the fit.
        """

        # Compute means
        x_mean = x.mean()
        y_mean = y.mean()

        # Compute slope m = Cov(x,y)/Var(x)
        # Cov = mean(x*y) - mean(x)*mean(y)
        cov_xy = (x * y).mean() - x_mean * y_mean
        var_x = (x * x).mean() - x_mean**2
        m = cov_xy / var_x

        # Compute intercept c = mean(y) - m*mean(x)
        c = y_mean - m * x_mean

        # Compute residuals and RMSE error term
        y_pred = m * x + c
        residuals = y - y_pred
        e = np.sqrt((residuals**2).mean())

        return m, c, e

    def instance_wise_timeout(self, delta):

        # initialize tresholds with 0
        thresholds = np.ascontiguousarray(
            np.full((self.number_of_instances,), 0, dtype=np.int32)
        )
        for i in self.instance_idx:
            all_runtimes = self.sorted_rt[rt][i, np.arange(self.number_of_solvers)]
            actu_par_2 = np.where(all_runtimes >= 5000, 10000, all_runtimes)
            total_runtime = all_runtimes.sum()
            best_timeout = 0
            best_o_delta = 0
            for j in range(self.number_of_solvers):
                if j == 0:
                    continue
                threshold = self.sorted_rt[rt][i, j]
                current_runtime = np.clip(all_runtimes, None, threshold).sum()
                runtime_frac = current_runtime / total_runtime
                pred_par_2 = np.where(all_runtimes > threshold, threshold * 2, all_runtimes)
                if (threshold >= 5000):
                    pred_par_2 = np.where(all_runtimes >= 5000, 10000, all_runtimes)

                acc = 1 - (np.sum(np.abs(pred_par_2 - actu_par_2)) / total_runtime)

                o_delta = delta*acc + ((1-delta) * (1-runtime_frac))

                #print(pred_par_2)
                #print(f"runtime_frac: {runtime_frac}")
                #print(f"acc: {acc}")
                #print(f"o_delta: {o_delta}")

                if o_delta > best_o_delta:
                    best_o_delta = o_delta
                    best_timeout = j

            thresholds[i] = best_timeout

            # Update self.pred
            best_timeout = self.sorted_rt[rt][i, thresholds[i]]
            best_pred_par_2 = np.where(all_runtimes > best_timeout, best_timeout * 2, all_runtimes)
            if (best_timeout >= 5000):
                best_pred_par_2 = np.where(all_runtimes >= 5000, 10000, all_runtimes)
            for index, solver in enumerate(self.sorted_rt[idx][i, np.arange(self.number_of_solvers)]):
                self.pred[solver] += best_pred_par_2[index]

            #if (best_timeout < 5000 and thresholds[i] < 27):
            #    print(f"instance {i} has timeout {thresholds[i]}: {best_timeout}s")

        return thresholds

    def sample_result(self, thresholds, pred, measurement, best_score=0):
        m, c, error = self.linear_fit(pred, self.par_2_scores)

        normalized_pred = m * pred + c

        cross_acc = self.calc_cross_acc_2(self.par_2_scores, normalized_pred)

        rmse_stability, cross_acc_stability  = self.calc_stability(thresholds, pred, error)

        new_pred = 0
        used_rt_removed_solver = 0
        if self.with_remaining_mean:
            penalties = self.get_remaining_mean(thresholds)
            choosen_instances = 0
            for index, runtime_list in enumerate(self.sorted_rt[rt]):
                timeout = runtime_list[thresholds[index]]
                if thresholds[index] == 0:
                    continue
                elif timeout > self.rt_removed_solver[index]:
                    used_rt_removed_solver += self.rt_removed_solver[index]
                    new_pred += self.rt_removed_solver[index]
                else:
                    used_rt_removed_solver += timeout
                    new_pred += penalties[index]
                choosen_instances += 1
            # new_pred = new_pred / choosen_instances
            # TODO: calculate other solvers also only on the choosen instances
        else:
            for index, runtime_list in enumerate(self.sorted_rt[rt]):
                timeout = runtime_list[thresholds[index]]
                # is instance maxed out?
                if thresholds[index] == self.number_of_reduced_solvers:
                    # is solver runtime 5000?
                    used_rt_removed_solver += self.rt_removed_solver[index]
                    if self.rt_removed_solver[index] == 5000:
                        new_pred += 10000
                    else:
                        new_pred += self.rt_removed_solver[index]
                # will a^ solve the instance
                elif timeout > self.rt_removed_solver[index]:
                    used_rt_removed_solver += self.rt_removed_solver[index]
                    new_pred += self.rt_removed_solver[index]
                # will a^ not solve the instance
                else:
                    used_rt_removed_solver += timeout
                    new_pred += 2 * timeout
        all_par_2_scores = np.append(
            self.par_2_scores, self.par_2_score_removed_solver
        )
        all_pred_use_known_par_2 = np.append(self.par_2_scores, m * new_pred + c)
        actual_all_pred = np.append(pred, new_pred)
        true_acc_v2 = self.calc_true_acc_1(
            all_par_2_scores,
            all_pred_use_known_par_2,
            self.number_of_reduced_solvers
        )
        true_acc_v1 = self.calc_true_acc_1(
            all_par_2_scores,
            actual_all_pred,
            self.number_of_reduced_solvers
        )

        runtime_frac = used_rt_removed_solver / self.total_rt_removed_solver
        #print(f"actual key  is  {self.pred_vec_to_key(all_par_2_scores)}")
        #print(f"pred key v1 is  {self.pred_vec_to_key(actual_all_pred)}")
        #print(f"pred key v2 is  {self.pred_vec_to_key(all_pred_use_known_par_2)}")
        print(f"best rmse is {error}")
        print(f"stability of this is {rmse_stability}")
        print(f"cross acc is {cross_acc}")
        print(f"stability of this is {cross_acc_stability}")
        print(f"with this, the new total is {used_rt_removed_solver} giving a fraction of {runtime_frac}")
        print(f"true acc v1 is {true_acc_v1}")
        print(f"true acc v2 is {true_acc_v2}")
        self.all_results.append({
                "solver": self.solver_string,
                "runtime_fraction": runtime_frac,
                "measurement": f"{measurement}_{self.thresh_breaking_condition.name}_true_acc_v1",
                "value": true_acc_v1
        })
        self.all_results.append({
                "solver": self.solver_string,
                "runtime_fraction": runtime_frac,
                "measurement": f"{measurement}_{self.thresh_breaking_condition.name}_true_acc_v2",
                "value": true_acc_v2
        })
        self.all_results.append({
                "solver": self.solver_string,
                "runtime_fraction": runtime_frac,
                "measurement": f"{measurement}_{self.thresh_breaking_condition.name}_cross_acc",
                "value": cross_acc
        })
        self.all_results.append({
                "solver": self.solver_string,
                "runtime_fraction": runtime_frac,
                "measurement": f"{measurement}_{self.thresh_breaking_condition.name}_diff",
                "value": error
        })
        self.all_results.append({
                "solver": self.solver_string,
                "runtime_fraction": runtime_frac,
                "measurement": f"{measurement}_{self.thresh_breaking_condition.name}_rmse_stability",
                "value": rmse_stability
        })
        self.all_results.append({
                "solver": self.solver_string,
                "runtime_fraction": runtime_frac,
                "measurement": f"{measurement}_{self.thresh_breaking_condition.name}_cross_acc_stability",
                "value": cross_acc_stability
        })
        return runtime_frac, cross_acc, cross_acc_stability

    def calc_stability(self, thresholds, pred, error):
        # current solver + its rt bearly solving the instance
        current_solver = self.sorted_rt[idx][self.instance_idx, thresholds], self.sorted_rt[rt][self.instance_idx, thresholds]
        current_penalty = current_solver[rt] * 2

        #_instance = 24
        #print(f"looking at instance {_instance}")
        #print("runtimes are:")
        #print(self.sorted_rt[rt][_instance])

        #print("solvers are:")
        #print(self.sorted_rt[idx][_instance])

        #print(f"current theshold is {thresholds[_instance]}")

        #print(f"this means a runtime of {self.sorted_rt[rt][_instance, thresholds[_instance]]}")

        # build a mask of which solvers timeout
        index_mask = np.arange(self.number_of_solvers)[None, :] > thresholds[:, None]

        timeout_times = self.sorted_rt[rt][self.instance_idx] * index_mask

        new_punish = np.where(timeout_times >= 5000.0, 10000.0, timeout_times)

        old_punish = current_penalty[:, None] * index_mask

        #print("the new punishment is")

        #print(new_punish[_instance])

        #print("the old punishment was")

        #print(old_punish[_instance])

        delta = new_punish - old_punish

        # Slice the -1 solver
        delta = delta[:, 1:]

        idx_chaos = self.sorted_rt[idx][:, 1:]

        ordered_delta = np.empty_like(delta)

        row_indices = np.arange(self.number_of_instances)[:, np.newaxis]
        ordered_delta[row_indices, idx_chaos] = delta   

        #print("delta is")
        #print(delta[_instance])

        #print("idx is")
        #print(idx_chaos[_instance])

        #print("ordered delta is")
        #print(ordered_delta[_instance])

        # copy previos pred to all instances
        new_pred = np.tile(pred, (self.number_of_instances, 1))

        #print("previous pred was")
        #print(pred)

        new_pred += ordered_delta

        #print("new perd is")
        #print(new_pred[_instance])

        similarity = batch_rmse(new_pred, self.par_2_scores)
        cross_acc = calc_cross_acc_batch(self.par_2_scores, new_pred)

        return similarity.mean(), cross_acc.mean()

    def sub_optimal_acc_maxing(self, new_acc: float, prev_acc: float):
        return new_acc <= prev_acc

    def sub_optimal_diff_mining(self, new_diff: float, prev_min: float):
        return new_diff >= prev_min

    def find_best_index_min_diff(
            self,
            thresholds,
            sorted_rt,
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
            thresholds,
            runtimes,
            actu_par_2,
            mean_actu: float,
            runtime_to_add: float,
            allowed_idxs=None,
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
        accs = concordant_count / self.reduced_solver_pairs        # (5355,)
        best_acc = accs[valid_mask].max()

        # 6) pick all indices within tol of the max
        best_mask = np.isclose(accs, best_acc, atol=tol)
        if allowed_idxs is None:
            best_idx = np.where(best_mask & valid_mask)[0]
        else:
            best_idx = allowed_idxs[best_mask & valid_mask]

        return best_idx, best_acc

    def is_new_sum_correct(self, sum, new_scores_adding_thresh_to_every_instance, thresholds, old_scores):
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

    def replace_by_overflow_mean(self, values, limits):
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
            thresholds,
            runtimes,
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
            thresholds,
            runtimes,
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
            thresholds,
            runtimes,
            actu
    ):
        pred = self.vec_to_pred(thresholds, runtimes)
        print(f"actual key is {self.pred_vec_to_key(actu)}")
        print(f"pred key is   {self.pred_vec_to_key(pred)}")

    def vec_to_diff(
            self,
            thresholds,
            runtimes,
            true_par_2,
            true_par_2_mean: float,
            removed_index: int
    ):
        pred_par_2 = np.delete(self.vec_to_pred(thresholds, runtimes), removed_index)

        return self.similarity(true_par_2, pred_par_2, true_par_2_mean)

    def vec_to_cross_acc(
            self,
            thresholds,
            runtimes,
            true_par_2
    ):
        pred_par_2 = self.vec_to_pred(thresholds, runtimes, 10000)

        acc = self.calc_cross_acc_2(true_par_2, pred_par_2)

        return acc

    def vec_to_true_acc(
            self,
            thresholds,
            runtimes,
            true_par_2,
            index: int
    ):
        pred_par_2_for_this_solver = self.vec_to_pred(thresholds, runtimes, 10000)[index]

        pred_par_2 = true_par_2.copy()
        pred_par_2[index] = pred_par_2_for_this_solver
        return self.calc_true_acc_1(true_par_2, pred_par_2, index)

    def vec_to_true_acc_2(
            self,
            thresholds,
            runtimes,
            true_par_2,
            index: int      
    ):
        scores = self.replace_by_overflow_mean(runtimes, thresholds)

    def determine_acc(self, actu, pred):
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

    def pred_vec_to_key(self, pred):
        """
        Sorts a 1D C‑contiguous float array `pred` (len ≤ 28) and returns
        a string of letters corresponding to the original indices in ascending order.
        returns None if any values in `pred` are exactly duplicated.
        """
        # 1) get sorted indices & sorted values
        idx = np.argsort(pred, kind=None)
        sv = pred[idx]

        # 2) detect duplicates in one vectorized call
        if np.any(sv[1:] == sv[:-1]):
            return None

        # 3) map sorted original‐indices → ASCII codes → bytes → string
        letters_u8 = self._letters[idx]
        return bytes(letters_u8).decode('ascii')

    def similarity(self, actu, pred, mean_actu: float):
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
            self, actu, pred
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

    def calc_cross_acc_3(self, actu, pred):
        # Compute pairwise differences
        pred_diff = pred[:, None] - pred
        true_diff = actu[:, None] - actu

        # Check for agreement in direction
        agreement = (pred_diff * true_diff) > 0

        # Sum up agreements per row and normalize
        rank_accuracies = agreement.sum(axis=1) / (actu.size - 1)

        return rank_accuracies.mean()

    def calc_true_acc_1(self, actu, pred, index: int):
        acc = 0
        pred_index = pred[index]
        actu_index = actu[index]
        solvers = actu.size
        for i in range(solvers):
            if (pred[i] - pred_index) * (actu[i] - actu_index) > 0:
                acc += 1
        return acc / (solvers - 1)

    def calc_true_acc_2(self, actu, pred, index: int) -> float:
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


def select_best_idx(score, remaining_mask, instance_idx):
    best_idx = np.nanargmax(score[remaining_mask])
    best_idx = instance_idx[remaining_mask][best_idx]
    return best_idx


def create_softmax_fn(temp):
    def select_idx_softmax(score, remaining_mask, instance_idx):
        valid_scores = score[remaining_mask]
        valid_scores = valid_scores - np.min(valid_scores)
        """ print("scores:")
        for sc in valid_scores:
            print(sc, end=", ")
        print() """
        std_dev = np.std(valid_scores)
        if std_dev == 0:
            std_dev = 1
        tau = std_dev * temp
        weights = np.power(np.e, -np.divide(valid_scores, tau))
        """ for i, w in enumerate(weights):
            #print(w, end=", ")
            if w == 0.0:
                print("weigth is zero !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print(f"delta score at this point is {valid_scores[i]}") """

        probabilities = weights / np.sum(weights)

        """ for p in probabilities:
            #print(w, end=", ")
            if p == 0.0:
                print("propability is zero !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        """
        # Get the indices of the valid scores
        valid_indices = np.arange(len(valid_scores))

        chosen_idx = int(np.random.choice(valid_indices, size=1, p=probabilities)[0])

        #print(f"choose instance having delta_score={valid_scores[chosen_idx]} and weight={weights[chosen_idx]}")

        chosen_idx = instance_idx[remaining_mask][chosen_idx]

        #print(f"verify instance having delta_score={(score - np.min(score))[chosen_idx]} and weight={np.e ** (-((score - np.min(score))[chosen_idx]/tau))}")

        return chosen_idx
    return select_idx_softmax


def create_top_k_sampling(k):
    def top_k_sampling(score, remaining_mask, instance_idx):
        # Get scores and indices of the remaining instances
        remaining_scores = score[remaining_mask]
        remaining_indices = instance_idx[remaining_mask]

        n = len(remaining_scores)

        if n == 0:
            return None  # or raise an error, depending on your use case

        if n <= k:
            # Not enough elements for top-k → pick randomly among all
            chosen_local = np.random.randint(0, n)
        else:
            # Get indices of the top-k scores
            # TODO: test if this line actually looks at the best k scores or not!
            top_k_local = np.argpartition(-remaining_scores, k-1)[:k]
            # Pick one of the top-k at random
            chosen_local = np.random.choice(top_k_local)

        # Map back to global index
        return remaining_indices[chosen_local]

    return top_k_sampling


def batch_rmse(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    For each row X[i,:], fit y ~ m*X[i,:] + c by least squares,
    and return the RMSE of that fit.
    Inputs:
      X: shape (5355, 27)
      y: shape (27,)
    Returns:
      e: shape (5355,)     -- e[i] is the RMSE for row i
    """
    # 1) compute per‑row slope and intercept
    x_mean = X.mean(axis=1)            # (N,)
    xx_mean = (X*X).mean(axis=1)        # (N,)
    xy_mean = (X*y).mean(axis=1)        # (N,)
    y_mean = y.mean()                  # scalar
    cov_xy = xy_mean - x_mean*y_mean    # (N,)
    var_x = xx_mean - x_mean**2       # (N,)
    m = cov_xy / var_x                  # (N,)
    c = y_mean - m*x_mean               # (N,)
    # detect zero‐variance rows
    zero_var = var_x == 0.0              # boolean mask shape (N,)
    if np.any(zero_var):
        # override those rows to horizontal fit
        m[zero_var] = 0.0
        c[zero_var] = y_mean
    # 2) build residuals and compute MSE directly
    #    `m[:,None]` broadcasts to shape (N,27)
    preds = m[:, None] * X + c[:, None]   # (N,27)
    residual = preds - y                     # (N,27)
    mse = (residual*residual).mean(axis=1)  # (N,)
    # 3) RMSE
    e = np.sqrt(mse)
    bad_indices = np.where(np.isneginf(e) | np.isinf(e) | np.isnan(e))[0]
    for bad_idx in bad_indices:
        print(f"found bad idx: {bad_idx}")
        print("pred:")
        print(X[bad_idx])
        print("mean")
        print(x_mean[bad_idx])
        print("xx_mean")
        print(xx_mean[bad_idx])
        print("xy_mean")
        print(xy_mean[bad_idx])
        print("cov_xy")
        print(cov_xy[bad_idx])
        print("var_x")
        print(var_x[bad_idx])
        print("m")
        print(m[bad_idx])
        print("c")
        print(c[bad_idx])
        print("mse")
        print(mse[bad_idx])
        print("e")
        print(e[bad_idx])
    return e


def calc_cross_acc_batch(actu: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """
    Compute "cross_acc" for each row of `pred` against `actu`.

    Parameters
    ----------
    actu : (N,) array_like
        Ground-truth values.
    pred : (M, N) array_like
        Each row is a prediction vector to compare with `actu`.

    Returns
    -------
    (M,) ndarray of float
        For each row m of `pred`, the fraction of concordant ordered pairs
        relative to `actu`: (# {(i,j): (pred_m[i]-pred_m[j]) and (actu[i]-actu[j]) have same nonzero sign})
        divided by N*(N-1). Ties count as non-concordant (same as your original function).
    """
    #actu = np.asarray(actu)
    #pred = np.asarray(pred)

    n = actu.size

    # Sign matrix for ground truth pairwise differences: shape (N, N)
    # Using int8 saves memory vs float.
    da_sign = np.sign(np.subtract.outer(actu, actu)).astype(np.int8)

    # Sign cubes for each prediction row: shape (M, N, N)
    dp_sign = np.sign(pred[:, :, None] - pred[:, None, :]).astype(np.int8)

    # Concordant where signs match and are nonzero
    mask = (dp_sign == da_sign) & (dp_sign != 0)

    # Count concordant pairs per row (exclude diagonal automatically, as sign=0 there)
    concordant = np.count_nonzero(mask, axis=(1, 2))

    denom = n * (n - 1)
    return concordant / denom


def greedy_rmse(new_pred, par_2_scores, total_added_runtime, rt_weight):
    rmse = batch_rmse(new_pred, par_2_scores) 
    return 1 / rmse


def knapsack_rmse(new_pred, par_2_scores, total_added_runtime, rt_weight):
    # Max 450
    rmse = batch_rmse(new_pred, par_2_scores)
    score = 135000000 / np.float_power(rmse, rt_weight)
    profitability_index = score / total_added_runtime  # similarity + self.rt_weight * total_added_runtime
    return profitability_index


def greedy_cross_acc(new_pred, par_2_scores, total_added_runtime, rt_weight):
    return calc_cross_acc_batch(par_2_scores, new_pred)


def knapsack_cross_acc(new_pred, par_2_scores, total_added_runtime, rt_weight):
    # Max 450
    cross_acc = calc_cross_acc_batch(par_2_scores, new_pred)
    profitability_index = cross_acc / total_added_runtime  # similarity + self.rt_weight * total_added_runtime
    return profitability_index


def create_cross_acc_breaking(break_at):
    def cross_acc_breaking(runtime_frac, cross_acc, stability):
        return cross_acc >= break_at
    return cross_acc_breaking


def create_stability_breaking(break_at):
    def cross_acc_breaking(runtime_frac, cross_acc, stability):
        return stability >= break_at
    return cross_acc_breaking
