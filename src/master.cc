#include "master.h"

namespace admm {

bool Master::GlobalUpdate(const std::vector<Master::real_t>& workers, AdmmConfig& admm_params, int num_worker) {
    std::size_t dim = workers.size();
    real_t crit = admm_params.global_var/admm_params.step_size;

    for (auto i = 0u; i < dim; ++i) {
        if (workers[i] > crit) {
            admm_params.global_weights[i] = (workers[i] - crit)/num_worker;
        } else if (workers[i] < -crit) {
            admm_params.global_weights[i] = (workers[i] + crit)/num_worker;
        } else {
            admm_params.global_weights[i] = 0;
        }
    }
    return true;
}
} // namespace admm
