#include "master.h"

namespace admm {

bool Master::GlobalUpdate(const std::vector<Master::Vec>& workers, AdmmConfig& admm_params) {
    std::size_t num_worker = workers.size();
    if (num_worker <= 0) {
        LOG(ERROR) << "None data is collected.";
        return false;
    }
    std::size_t dim = workers[0].size();
    if (dim <= 0) {
        LOG(ERROR) << "Gathered vectors are empty.";
        return false;
    }

    for (auto i = 0u; i < dim; ++i) {
        real_t sum = 0;
        for(auto k = 0u; k < num_worker; ++k) {
            sum += workers[k][i]; 
        }
        real_t crit = admm_params.global_var/admm_params.step_size;
        if (sum > crit) {
            admm_params.global_weights[i] = (sum - crit)/num_worker;
        } else if (sum < -crit) {
            admm_params.global_weights[i] = (sum + crit)/num_worker;
        } else {
            admm_params.global_weights[i] = 0;
        }
    }
    return true;
}
} // namespace admm
