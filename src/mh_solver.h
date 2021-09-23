#include "common.h"
#include "solver.h"

class MetropolisHastings : public Solver {

    MatrixXd _initial_thetas;
    VectorXd _thetas;
    std::vector<double> _thetas_lookup;

    public:
        MetropolisHastings(PointMatrix& pts, 
                int min_bound, 
                int max_bound, 
                int min_dist,
                int seed,
                MatrixXd counts,
                MatrixXd dcounts,
                MatrixXd initial_thetas,
                long long int n_iters,
                string output_dir,
                long long int output_every,
                bool output_txt,
                long long int rate_every,
                int print_only_first,
				bool shuffle) :                 
            Solver(
                pts,
                min_bound,
                max_bound,
                min_dist,
                seed,
                counts,
                dcounts,
                n_iters,
                output_dir,
                output_every,
                output_txt,
                rate_every,
                print_only_first,
                shuffle
            ),
            _initial_thetas(initial_thetas),
            _thetas(_n_bins)
            {
            
                // Initialize the thetas
                for (int n_bin = 0; n_bin < _n_bins; n_bin++) {
                    // If initial thetas not given, use "ratio"
                    if (_initial_thetas.rows() == 0) {
                        _thetas(n_bin) = max(
                            log(_counts_table(n_bin, 1)) - log(_dcounts_table(n_bin, 1)), 
                            static_cast<double>(VERY_SMALL_LOG));

                        if (isnan(_thetas(n_bin))) {
                            _thetas(n_bin) = VERY_SMALL_LOG;
                        }
                    } 
                    else {
                        _thetas(n_bin) = _initial_thetas(n_bin, 1);                        
                    }
                }
                

                // Create a fast lookup 
                int distance = 0;

                int histogram_index = 1;
                double max_distance = ceil(_pts.maxCoeff() - _pts.minCoeff() + 1);
                while (distance < max_distance) {
                    if (distance >= _histogram_ticks[histogram_index]) {
                        histogram_index++;
                        if (histogram_index == _histogram_ticks.size()) {
                            while (distance < max_distance) {
                                _thetas_lookup.push_back(VERY_SMALL_LOG);
                                distance++;
                            }
                            break;
                        }
                    }
                    if (histogram_index < _histogram_values.size()) {
                        _thetas_lookup.push_back(_thetas[histogram_index-1]);            
                    } else {
                        _thetas_lookup.push_back(VERY_SMALL_LOG);
                    }
                    distance++;
                }
            }

    virtual double compute_log_ratio(TwoContactsMatrix& two_old_contacts, TwoContactsMatrix& two_new_contacts) {
        long old_first_distance = two_old_contacts(0, 1) - two_old_contacts(0, 0);
        long old_second_distance = two_old_contacts(1, 1) - two_old_contacts(1, 0);
        long new_first_distance = two_new_contacts(0, 1) - two_new_contacts(0, 0);
        long new_second_distance = two_new_contacts(1, 1) - two_new_contacts(1, 0);

        double ret = 0.0;

        /*
        ret += _histogram_lookup[new_first_distance];
        ret -= log(_dcounts_table(_bin_lookup[new_first_distance], 1));

        ret -= _histogram_lookup[old_first_distance];
        ret += log(_dcounts_table(_bin_lookup[old_first_distance], 1));

        ret += _histogram_lookup[new_second_distance];
        ret -= log(_dcounts_table(_bin_lookup[new_second_distance], 1));

        ret -= _histogram_lookup[old_second_distance];
        ret += log(_dcounts_table(_bin_lookup[old_second_distance], 1));
        */
        ret += _thetas_lookup[new_first_distance];
        ret -= _thetas_lookup[old_first_distance];
        ret += _thetas_lookup[new_second_distance];
        ret -= _thetas_lookup[old_second_distance];

        return ret;
    }            

    virtual bool decide_to_accept(double accept_p) { 
        return (_unit_interval_sampler(_g) <= accept_p);
    }
};
