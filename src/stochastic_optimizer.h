#include "common.h"
#include "solver.h"

class Adam {
    public:
        int _n_parameters;
        double _alpha;
        double _beta1;
        double _beta2;
        double _epsilon;

        VectorXd _m;
        VectorXd _v;    
        VectorXd _mub;
        VectorXd _vub;    

        long long int _n_iter;    

        Adam(
            int n_parameters,
            double alpha = 0.001,
            double beta1 = 0.9,
            double beta2 = 0.999,
            double epsilon = 1e-8
            ) :
            _n_parameters(n_parameters),
            _alpha(alpha),
            _beta1(beta1),
            _beta2(beta2),
            _epsilon(epsilon),
            _m(VectorXd::Zero(_n_parameters)),
            _v(VectorXd::Zero(_n_parameters)),
            _n_iter(0)
            {

            }

        void calculate(const VectorXd& gradient, VectorXd& output) {
            _n_iter++;

            _m = _beta1 * _m + (1 - _beta1) * gradient;
            _mub = _m / (1 - pow(_beta1, _n_iter));

            _v = _beta2 * _v + (1 - _beta2) * gradient.array().pow(2).matrix();
            _vub = _v / (1 - pow(_beta2, _n_iter));

            output = _alpha * (_mub.array() / (_vub.array().sqrt() + _epsilon)).matrix();            
        }

};


class StochasticOptimizer : public Solver {
    public:

        long long int _srf_every;
        long long int _srf_n0;
        long long int _srf_n_iter;
        double _srf_exponent;
        double _srf_multiplier;
        double _sgd_smoothness_regularization_penalty;
        double _sgd_closeness_regularization_penalty;

        double _beta1;

        MatrixXd _curr_inv_jacobian;

        VectorXd _thetas;
        VectorXd _thetas_star;
        VectorXd _target_histogram;
        VectorXd _probs_histogram;
        VectorXd _ones;
        VectorXd _drift_z;
        VectorXd _smoothed_p;

        ofstream _thetas_file;
        ofstream _deltas_file;
        ofstream _histograms_file;

        Adam _adam;

        StochasticOptimizer(PointMatrix& pts, 
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
                double srf_multiplier,
                double sgd_smoothness_regularization_penalty,
                double sgd_closeness_regularization_penalty
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
            _sgd_smoothness_regularization_penalty(sgd_smoothness_regularization_penalty),
            _sgd_closeness_regularization_penalty(sgd_closeness_regularization_penalty),
            _beta1(0.9),
            _thetas(_n_bins),
            _target_histogram(_n_bins),
            _probs_histogram(_n_bins),
            _ones(VectorXd::Ones(_n_bins)),
            _drift_z(_n_bins),
            _smoothed_p(VectorXd::Zero(_n_bins)),
            _adam(_n_bins)
            {
                for (int n_bin = 0; n_bin < _n_bins; n_bin++) {
                    _target_histogram(n_bin) = _counts_table(n_bin, 1);

                    _thetas(n_bin) = max(
                        log(_counts_table(n_bin, 1)) - log(_dcounts_table(n_bin, 1)), 
                        static_cast<double>(VERY_SMALL_LOG));

                    if (isnan(_thetas(n_bin))) {
                        _thetas(n_bin) = VERY_SMALL_LOG;
                    }

                    _drift_z(n_bin) = n_bin;
                }

                _thetas_star = _thetas;

                for (int n_bin = 0; n_bin < _n_bins; n_bin++) {
                    _drift_z(n_bin) -= (_drift_z.sum() / _n_bins);
                }

                _thetas_file.open(
                    (boost::filesystem::path(_output_dir) / boost::filesystem::path("output." + to_string(_n_iters) + ".thetas")).string()
                    );


                _histograms_file.open(
                    (boost::filesystem::path(_output_dir) / boost::filesystem::path("output." + to_string(_n_iters) + ".hists")).string()
                    );   

                _deltas_file.open(
                    (boost::filesystem::path(_output_dir) / boost::filesystem::path("output." + to_string(_n_iters) + ".deltas")).string()
                    );                     
            }


    void multiply_jacobian_by_vector(const VectorXd& pvec, const VectorXd& v, VectorXd& reply) {
        reply = (pvec.array() * v.array()).matrix();
        reply = reply - pvec.dot(v) * pvec;
        reply = _n_points * reply;
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

    virtual void smooth_regularization_gradient(const VectorXd& thetas, VectorXd& reply) {
        // First and last are exempt
        reply[0] = 0;
        reply[_n_bins-1] = 0;
        
        reply[1] = thetas[1] - thetas[2];
        reply[_n_bins-2] = thetas[_n_bins-2] - thetas[_n_bins-3];
        for (int i=2; i < _n_bins-2; i++) {
            reply[i] = 2*thetas[i] - thetas[i-1] - thetas[i+1];
        }
    }

    virtual void star_regularization_gradient(const VectorXd& thetas, VectorXd& reply) {
        for (int i=0; i < _n_bins; i++) {
            reply[i] = thetas[i] - _thetas_star[i];
        }
    }    

    virtual void loss_drift(const VectorXd& pvec, const VectorXd& g_minus_d, VectorXd& reply) {
        multiply_jacobian_by_vector(pvec, _drift_z, reply);
        reply *= _drift_z.dot(g_minus_d);
        // TODO: precalc
        reply /= (_drift_z.squaredNorm() * _drift_z.squaredNorm());
    }    


    virtual void post_step(long long int n_iter) {
        if ((n_iter % _srf_every) == 0) {
            // Stochastic approximation step
            //double gain = 1.0 / pow(max(static_cast<long long int>(1), _srf_n_iter + _srf_n0), _srf_exponent);

            VectorXd delta(_n_bins);
            VectorXd drift_delta(_n_bins);
            VectorXd smooth_reg_delta(_n_bins);
            VectorXd star_reg_delta(_n_bins);

            _smoothed_p = _beta1 * _smoothed_p + (1 - _beta1) * (_current_histogram.cast<double>() / _current_histogram.cast<double>().sum());

            multiply_jacobian_by_vector(
                //_target_histogram.cast<double>() / _target_histogram.cast<double>().sum(), 
                //_current_histogram.cast<double>() / _current_histogram.cast<double>().sum(),
                _smoothed_p / (1 - pow(_beta1, (_srf_n_iter+1))),
                (_current_histogram.cast<double>() - _target_histogram),
                delta
                );
            
            //delta *= _srf_multiplier;

            // loss_drift(
            //     _current_histogram.cast<double>() / _current_histogram.cast<double>().sum(), 
            //     (_current_histogram.cast<double>() - _target_histogram),
            //     drift_delta
            // );

            smooth_regularization_gradient(_thetas, smooth_reg_delta);
            star_regularization_gradient(_thetas, star_reg_delta);

            
            // _thetas = _thetas - _srf_multiplier * gain * (
            //     delta + 
            //     100 * _sgd_regularization_penalty * star_reg_delta
            //     //_sgd_regularization_penalty * smooth_reg_delta + 
            //     //_sgd_regularization_penalty * drift_delta
            //     );

            VectorXd adam_output(_n_bins);

            _adam.calculate(
                delta + _sgd_closeness_regularization_penalty * star_reg_delta + _sgd_smoothness_regularization_penalty * smooth_reg_delta,
                adam_output
            );

            _thetas = _thetas - _srf_multiplier * adam_output;               


            if ((n_iter % _rate_every) == 0) {
                _thetas_file << _thetas.transpose().format(Eigen::IOFormat(Eigen::FullPrecision)) << endl;
                _histograms_file << _current_histogram.transpose() << endl;
                _deltas_file << _adam._v.transpose().format(Eigen::IOFormat(Eigen::FullPrecision)) << endl;
            }
            
            _srf_n_iter++;
        }
    }
};
