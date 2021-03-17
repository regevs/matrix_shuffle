#include "common.h"
#include "solver.h"

class GreedyOptimizer : public Solver {
    public:
        GreedyOptimizer(PointMatrix& pts, 
                int min_bound, 
                int max_bound, 
                int min_dist,
                int seed,
                MatrixXd counts,
                MatrixXd dcounts,
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
            ) {}

    virtual double compute_log_ratio(TwoContactsMatrix& two_old_contacts, TwoContactsMatrix& two_new_contacts) {
        long old_first_distance = two_old_contacts(0, 1) - two_old_contacts(0, 0);
        long old_second_distance = two_old_contacts(1, 1) - two_old_contacts(1, 0);
        long new_first_distance = two_new_contacts(0, 1) - two_new_contacts(0, 0);
        long new_second_distance = two_new_contacts(1, 1) - two_new_contacts(1, 0);

        double ret = 0.0;

        ret += _histogram_lookup[new_first_distance];
        ret -= log(_current_histogram[_bin_lookup[new_first_distance]]+1);
        
        ret -= _histogram_lookup[old_first_distance];
        ret += log(_current_histogram[_bin_lookup[old_first_distance]]);

        ret += _histogram_lookup[new_second_distance];
        ret -= log(_current_histogram[_bin_lookup[new_second_distance]]+1);

        ret -= _histogram_lookup[old_second_distance];
        ret += log(_current_histogram[_bin_lookup[old_second_distance]]);

        return ret;
    }            

    virtual bool decide_to_accept(double accept_p) { 
        return (accept_p >= 1); 
    }
};

class MetropolisHastings : public Solver {
    public:
        MetropolisHastings(PointMatrix& pts, 
                int min_bound, 
                int max_bound, 
                int min_dist,
                int seed,
                MatrixXd counts,
                MatrixXd dcounts,
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
            ) {}

    virtual double compute_log_ratio(TwoContactsMatrix& two_old_contacts, TwoContactsMatrix& two_new_contacts) {
        long old_first_distance = two_old_contacts(0, 1) - two_old_contacts(0, 0);
        long old_second_distance = two_old_contacts(1, 1) - two_old_contacts(1, 0);
        long new_first_distance = two_new_contacts(0, 1) - two_new_contacts(0, 0);
        long new_second_distance = two_new_contacts(1, 1) - two_new_contacts(1, 0);

        double ret = 0.0;

        ret += _histogram_lookup[new_first_distance];
        ret -= log(_dcounts_table(_bin_lookup[new_first_distance], 1));

        ret -= _histogram_lookup[old_first_distance];
        ret += log(_dcounts_table(_bin_lookup[old_first_distance], 1));

        ret += _histogram_lookup[new_second_distance];
        ret -= log(_dcounts_table(_bin_lookup[new_second_distance], 1));

        ret -= _histogram_lookup[old_second_distance];
        ret += log(_dcounts_table(_bin_lookup[old_second_distance], 1));

        return ret;
    }            

    virtual bool decide_to_accept(double accept_p) { 
        return (_unit_interval_sampler(_g) <= accept_p);
    }
};
