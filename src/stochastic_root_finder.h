#include "common.h"
#include "solver.h"

class StochasticRootFinder : public Solver {
    public:

        long long int _srf_every;
        long long int _srf_n0;
        long long int _srf_n_iter;
        double _srf_exponent;
        double _srf_multiplier;

        MatrixXd _curr_inv_jacobian;

        VectorXd _thetas;
        VectorXd _target_histogram;
        VectorXd _probs_histogram;
        VectorXd _ones;

        ofstream _thetas_file;
        ofstream _deltas_file;
        ofstream _histograms_file;

        StochasticRootFinder(PointMatrix& pts, 
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
				bool shuffle,
                long long int srf_every,
                long long int srf_n0,
                double srf_exponent,
                double srf_multiplier
                ) : 
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
            _srf_every(srf_every),
            _srf_n0(srf_n0),
            _srf_n_iter(0),
            _srf_exponent(srf_exponent),
            _srf_multiplier(srf_multiplier),
            _thetas(_n_bins),
            _target_histogram(_n_bins),
            _probs_histogram(_n_bins),
            _ones(VectorXd::Ones(_n_bins))
            {
                for (int n_bin = 0; n_bin < _n_bins; n_bin++) {
                    _target_histogram(n_bin) = _counts_table(n_bin, 1);

                    _thetas(n_bin) = max(
                        log(_counts_table(n_bin, 1)) - log(_dcounts_table(n_bin, 1)), 
                        static_cast<double>(VERY_SMALL_LOG));

                    if (isnan(_thetas(n_bin))) {
                        _thetas(n_bin) = VERY_SMALL_LOG;
                    }
                }

                // calculate_current_histogram();
                // update_inv_jacobian(_target_histogram);

                // chrono::steady_clock::time_point begin = chrono::steady_clock::now();
                // update_inv_jacobian();
                // chrono::steady_clock::time_point end = chrono::steady_clock::now();
                // cout << "Jac inv: " << chrono::duration_cast<chrono::seconds> (end - begin).count() << "[sec]" << endl;


                _thetas_file.open(
                    (boost::filesystem::path(_output_dir) / boost::filesystem::path("output." + to_string(_n_iters) + ".thetas")).string()
                    );

                // _deltas_file.open(
                //     (boost::filesystem::path(_output_dir) / boost::filesystem::path("output." + to_string(_n_iters) + ".deltas")).string()
                //     );

                _histograms_file.open(
                    (boost::filesystem::path(_output_dir) / boost::filesystem::path("output." + to_string(_n_iters) + ".hists")).string()
                    );                    
            }

    void update_inv_jacobian(const VectorXd& pvec) {
        //_probs_histogram = _target_histogram / _target_histogram.sum();
        _probs_histogram = pvec.cast<double>() / pvec.sum();

        MatrixXd diag(_probs_histogram.asDiagonal());
        _curr_inv_jacobian = _n_points * (diag - _probs_histogram * _probs_histogram.transpose());
        _curr_inv_jacobian = _curr_inv_jacobian.completeOrthogonalDecomposition().pseudoInverse();
    }

    void multiply_inv_jacobian_by_vector(const VectorXd& pvec, const VectorXd& v, VectorXd& reply) {
        VectorXd pvecinv = VectorXd::Zero(_n_bins);   // TODO - don't reallocate
        VectorXd z = VectorXd::Zero(_n_bins);   // TODO - don't reallocate
        int Bt = 0;
        
        for (int i = 0; i < _n_bins; i++) {
            if (pvec(i) > 0) {
                pvecinv(i) = 1/pvec(i);
                z(i) = 1;
                Bt++;
            }
        }

        reply = (v.array() * pvecinv.array()).matrix();
        reply = reply - (v.array() * pvecinv.array()).matrix().dot(z) / Bt * z;
        reply = reply - ((v.dot(z) * z).array() * pvecinv.array()).matrix() / Bt;
        reply = reply + pvecinv.sum() / (Bt * Bt) * v.dot(z) * z;
        reply = reply / _n_points;

        // reply = (v.array() * pvecinv.array()).matrix();
        // reply = reply - (v.array() * pvecinv.array()).sum() / _n_bins * _ones ;
        // reply = reply - (1.0 / _n_bins) * ((v.sum() * _ones).array() * pvecinv.array()).matrix();
        // reply = reply + pvecinv.sum() / (_n_bins * _n_bins) * v.sum() * _ones;
        // reply = reply / _n_points;
    }

    virtual double compute_log_ratio(TwoContactsMatrix& two_old_contacts, TwoContactsMatrix& two_new_contacts) {
        long old_first_distance = two_old_contacts(0, 1) - two_old_contacts(0, 0);
        long old_second_distance = two_old_contacts(1, 1) - two_old_contacts(1, 0);
        long new_first_distance = two_new_contacts(0, 1) - two_new_contacts(0, 0);
        long new_second_distance = two_new_contacts(1, 1) - two_new_contacts(1, 0);

        double ret = 0.0;

        ret += _thetas(_bin_lookup[new_first_distance]);
        ret -= _thetas(_bin_lookup[old_first_distance]);
        ret += _thetas(_bin_lookup[new_second_distance]);
        ret -= _thetas(_bin_lookup[old_second_distance]);

        return ret;
    }            

    virtual bool decide_to_accept(double accept_p) { 
        return (_unit_interval_sampler(_g) <= accept_p);
    }

    virtual void post_step(long long int n_iter) {
        if ((n_iter % _srf_every) == 0) {
            // Stochastic approximation step
            double gain = 1.0 / pow(max(static_cast<long long int>(1), _srf_n_iter + _srf_n0), _srf_exponent);

            VectorXd delta(_n_bins);

            // multiply_inv_jacobian_by_vector(
            //     _current_histogram.cast<double>() / _current_histogram.cast<double>().sum(), 
            //     (_current_histogram.cast<double>() - _target_histogram),
            //     delta
            //     );
            multiply_inv_jacobian_by_vector(
                _target_histogram.cast<double>() / _target_histogram.cast<double>().sum(), 
                (_current_histogram.cast<double>() - _target_histogram),
                delta
                );
            
            //delta =  _curr_inv_jacobian * (_current_histogram.cast<double>() - _target_histogram);
            delta *= _srf_multiplier;

            _thetas = _thetas - gain * delta;
                


            if ((n_iter % _rate_every) == 0) {
                _thetas_file << _thetas.transpose().format(Eigen::IOFormat(Eigen::FullPrecision)) << endl;
                _histograms_file << _current_histogram.transpose() << endl;
                //_deltas_file << delta.transpose().format(Eigen::IOFormat(Eigen::FullPrecision)) << endl;
            }
            
            _srf_n_iter++;
        }
    }
};