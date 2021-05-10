#pragma once
#include "common.h"
#include "io.h"

// Point holders
#include "point_holder.h"

mutex progress_bar_lock;
boost::progress_display* progress_bar;

class Solver {
  public:
    MatrixXd _counts_table;
    MatrixXd _dcounts_table;
    int _n_bins;
    
    std::vector<double> _histogram_ticks;
    std::vector<double> _histogram_values;

    std::vector<int> _bin_lookup;
    std::vector<double> _histogram_lookup;

    long long int _n_iters;
    int _min_bound;
    int _max_bound;
    int _min_dist;
    
    string _output_dir;
    long long int _output_every;
    bool _output_txt;
    long long int _rate_every;
    int _print_only_first;

    PointMatrix& _pts;
    int _n_points;    

    mt19937 _g;
    uniform_real_distribution<double> _unit_interval_sampler;
    std::uniform_int_distribution<> _uniform_point_sampler; 
    
    double _target_log_likelihood;
    double _starting_log_likelihood;

    VectorXi _current_histogram;
    
	long _l1_distance;
    bool _shuffle;

    boost::scoped_ptr<PointHolder> _point_holder;

    Solver(PointMatrix& pts, 
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
        _counts_table(counts),
        _dcounts_table(dcounts),
        _n_bins(_counts_table.rows() - 1),
        _n_iters(n_iters),
        _min_bound(min_bound),
        _max_bound(max_bound),
        _min_dist(min_dist),
        _output_dir(output_dir),
        _output_every(output_every),
        _output_txt(output_txt),
        _rate_every(rate_every),
        _print_only_first(print_only_first),
        _pts(pts),
        _n_points(_pts.rows()),
        _g(seed),
        _uniform_point_sampler(0, _n_points-1),
		_shuffle(shuffle)
    {
        cout << "Starting constructor... ";

        if (_print_only_first == -1) {
            _print_only_first = pts.rows();
        }
        
        for (int i = 0; i < counts.rows(); i++) {
            _histogram_ticks.push_back(counts(i, 0));
            _histogram_values.push_back(counts(i, 1));
        }

        int distance = 0;
        if (_histogram_ticks[0] > 0) {
            cerr << "First histogram tick should be 0." << endl;
            exit(-1);
        }

        int histogram_index = 1;
        double max_distance = ceil(pts.maxCoeff() - pts.minCoeff() + 1);
        while (distance < max_distance) {
            if (!(histogram_index < _histogram_ticks.size())) {
                cout << histogram_index << " " << _histogram_ticks.size() << endl;
                cout <<_histogram_ticks[0] << " " << _histogram_ticks[_histogram_ticks.size()-1] << endl;
                exit(-1);
            }
            if (distance >= _histogram_ticks[histogram_index]) {
                histogram_index++;
                if (histogram_index == _histogram_ticks.size()) {
                    while (distance < max_distance) {
                        _bin_lookup.push_back(_histogram_ticks.size() - 1);  // I hope this it true
                        _histogram_lookup.push_back(VERY_SMALL_LOG);
                        distance++;
                    }
                    break;
                }
            }
            if (histogram_index < _histogram_values.size()) {
                _bin_lookup.push_back(histogram_index-1);
//                 _histogram_lookup.push_back(log(_histogram_values[histogram_index-1]) - log(dcounts(histogram_index-1, 1)));            
                _histogram_lookup.push_back(log(_histogram_values[histogram_index-1]));            
            } else {
                _bin_lookup.push_back(_histogram_ticks.size()-1);
                _histogram_lookup.push_back(VERY_SMALL_LOG);
            }
            distance++;
        }

        cout << "Done." << endl;

    }
    
    virtual double compute_log_ratio(TwoContactsMatrix& two_old_contacts, TwoContactsMatrix& two_new_contacts) { return 0.0; };    

    uint32_t draw_random() {
        return _uniform_point_sampler(_g);
    }

    void draw_switch_uniformly(TwoContactsMatrix& two_old_contacts,
                     TwoContactsMatrix& two_new_contacts,
                     pair<int, int>& two_chosen_inds,
                     int first_chosen_index = -1) {
        
        if (first_chosen_index != -1) {
            two_chosen_inds.first = first_chosen_index;
        } else {
            two_chosen_inds.first = draw_random();
        }
        two_chosen_inds.second = draw_random();
                         
        if (two_chosen_inds.second < two_chosen_inds.first) {
            std::swap(two_chosen_inds.first, two_chosen_inds.second);            
        }

        two_old_contacts(0, 0) = _pts(two_chosen_inds.first, 0);
        two_old_contacts(0, 1) = _pts(two_chosen_inds.first, 1);
        two_old_contacts(1, 0) = _pts(two_chosen_inds.second, 0);
        two_old_contacts(1, 1) = _pts(two_chosen_inds.second, 1);

        // Flip a coin
        int which_side = (_unit_interval_sampler(_g) < 0.5);
        two_new_contacts(0, 0) = two_old_contacts(0, 0);
        two_new_contacts(0, 1) = two_old_contacts(1, which_side);
        two_new_contacts(1, 0) = two_old_contacts(which_side, 1-which_side);
        two_new_contacts(1, 1) = two_old_contacts(1-which_side, 1);
       
        for (int l = 0; l < 2; l++) {
            if (two_new_contacts(l, 0) > two_new_contacts(l, 1)) {
                std::swap(two_new_contacts(l, 0), two_new_contacts(l, 1));
            }
        }
    }

    void perform_switch_in_points(TwoContactsMatrix& two_new_contacts,
                        pair<int, int>& two_chosen_inds) {
        _pts.row(two_chosen_inds.first) = two_new_contacts.row(0);
        _pts.row(two_chosen_inds.second) = two_new_contacts.row(1);
    }

    void update_current_histogram(
        TwoContactsMatrix& two_old_contacts,
        TwoContactsMatrix& two_new_contacts) {
        // Update the current histogram
        long old_first_distance = two_old_contacts(0, 1) - two_old_contacts(0, 0);
        long old_second_distance = two_old_contacts(1, 1) - two_old_contacts(1, 0);
        long new_first_distance = two_new_contacts(0, 1) - two_new_contacts(0, 0);
        long new_second_distance = two_new_contacts(1, 1) - two_new_contacts(1, 0);

        assert(_current_histogram[_bin_lookup[old_first_distance]] > 0);
        assert(_current_histogram[_bin_lookup[old_second_distance]] > 0);

        // Update L1 distance (remove parts we're gonna change)
        _l1_distance -= abs(_current_histogram[_bin_lookup[old_first_distance]] - _histogram_values[_bin_lookup[old_first_distance]]);
        _l1_distance -= abs(_current_histogram[_bin_lookup[old_second_distance]] - _histogram_values[_bin_lookup[old_second_distance]]);
        _l1_distance -= abs(_current_histogram[_bin_lookup[new_first_distance]] - _histogram_values[_bin_lookup[new_first_distance]]);
        _l1_distance -= abs(_current_histogram[_bin_lookup[new_second_distance]] - _histogram_values[_bin_lookup[new_second_distance]]);

        _current_histogram[_bin_lookup[old_first_distance]]--;
        _current_histogram[_bin_lookup[old_second_distance]]--;
        _current_histogram[_bin_lookup[new_first_distance]]++;
        _current_histogram[_bin_lookup[new_second_distance]]++;

        // Update L1 distance (re-add them)
        _l1_distance += abs(_current_histogram[_bin_lookup[old_first_distance]] - _histogram_values[_bin_lookup[old_first_distance]]);
        _l1_distance += abs(_current_histogram[_bin_lookup[old_second_distance]] - _histogram_values[_bin_lookup[old_second_distance]]);
        _l1_distance += abs(_current_histogram[_bin_lookup[new_first_distance]] - _histogram_values[_bin_lookup[new_first_distance]]);
        _l1_distance += abs(_current_histogram[_bin_lookup[new_second_distance]] - _histogram_values[_bin_lookup[new_second_distance]]);
            
    }

    void shuffle_coordinates() {
        cerr << "Shuffling coordinates... ";

        std::shuffle(_pts.data(), _pts.data() + _pts.size(), _g);
        for (int i = 0; i < _n_points; i++) {
            if (_pts(i, 1) < _pts(i, 0)) {
                std::swap(_pts(i, 0), _pts(i, 1));
            }
        }

        cerr << "Done." << endl;
    }

    void verify_minimal_distance() {
        cerr << "Verifying minimal distance after shuffle... ";

        TwoContactsMatrix two_old_contacts, two_new_contacts; 
        pair<int, int> two_chosen_inds;
        int corrected = 0;

        for (int i = 0; i < _n_points; i++) {
            if (_pts(i, 1) - _pts(i, 0) < _min_dist) {
                corrected++;
                while (true) {
                    draw_switch_uniformly(two_old_contacts, two_new_contacts, two_chosen_inds, i);
                    if (((two_new_contacts(0, 1) - two_new_contacts(0, 0)) >= _min_dist) &&
                        ((two_new_contacts(1, 1) - two_new_contacts(1, 0)) >= _min_dist)) {
                        perform_switch_in_points(two_new_contacts, two_chosen_inds);
                        break;
                    }
                }
            }
        }

        cerr << "Done (" << corrected << " points corrected)." << endl;
        cerr << "First: " << endl;
        cerr << _pts.topRows(5).cast<int>() << endl << endl;
    }
    
    double log_factorial(int n) {
        return std::lgamma(n+1);
    }

    void calculate_target_likelihood() {
        _target_log_likelihood = log_factorial(_n_points);
        int n_i;

        for (int i = 0; i < _n_bins; i++) {
            n_i = _histogram_values[i];
            if (n_i == 0) {
                continue;
            }

            _target_log_likelihood += (n_i * (log(n_i) - log(_n_points)) - log_factorial(n_i));
        }

        cerr << "Target log likelihood: " << _target_log_likelihood << endl;
    }

    void calculate_current_histogram() {
        // Initialize and zero
        _current_histogram.resize(_n_bins);
        _current_histogram.setZero();

        // Count
        for (int n_point = 0; n_point < _n_points; n_point++) {
            _current_histogram[_bin_lookup[_pts(n_point, 1) - _pts(n_point, 0)]]++;
        }        

        // Initialize L1 distance
        _l1_distance = 0;
        for (int n_bin = 0; n_bin < _n_bins; n_bin++) {
            int diff = abs(_current_histogram[n_bin] - _histogram_values[n_bin]);
            _l1_distance += diff;            
        }
    }

    int calculate_max_abs_diff() {
        // Initialize 
        int _max_abs_diff = 0;
        for (int n_bin = 0; n_bin < _n_bins; n_bin++) {
            int diff = abs(_current_histogram[n_bin] - _histogram_values[n_bin]);
            if (diff > _max_abs_diff) {
                _max_abs_diff = diff;
            }
        }

        return _max_abs_diff;        
    }

    double calculate_current_likelihood() {
        double ll = log_factorial(_n_points);

        int n_i;

        for (int i = 0; i < _n_bins; i++) {
            n_i = _current_histogram[i];
            if (n_i == 0) {
                continue;
            }

            // In case one of the target histogram bins is empty, to avoid -inf log likelihood, 
            // replace log(d_i) with VERY_SMALL_LOG
            ll += (n_i * (std::max(double(VERY_SMALL_LOG), log(_histogram_values[i])) - log(_n_points)) - log_factorial(n_i));
        }

        return ll;
    }

    void calculate_starting_likelihood() {
        _starting_log_likelihood = calculate_current_likelihood();

        cerr << "Starting log likelihood: " << _starting_log_likelihood << endl;
    }

    virtual bool decide_to_accept(double accept_p) { return true; }
    
    virtual void post_step(long long int n_iter) { }

    void run() {
        //
        // Initialize
        //
        chrono::steady_clock::time_point begin = chrono::steady_clock::now();
        TwoContactsMatrix two_old_contacts, two_new_contacts; 
        pair<int, int> two_chosen_inds;        

        double log_ratio;
        double probability_of_new_given_old;
        double probability_of_old_given_new;
        double accept_p;

        long long int accept_count = 0;
        long long int accept_count_same_bin = 0;
        long long int total_same_bin = 0;
        long long int accept_count_general = 0;
        long long int total_general = 0;
        long long int total_accept_count = 0;
        long long int total_improved_delta = 0;

        //
        // Shuffle coordinates properly
        //
        if (_shuffle) {
            shuffle_coordinates();
            verify_minimal_distance();
        }

        //
        // Initialize point holder
        //

        // _point_holder.reset(new ShiftedPointHolder(
        //     _pts,
        //     _g,
        //     _max_bound / 2,
        //     _max_bound,
        //     int(_max_bound / 4),
        //     int(_max_bound / 4)
        // ));

        // MatrixXi shifts;
        // int W = int(_max_bound / 100);

        // shifts.resize(2, 2);
        // shifts <<
        //     0, 0,
        //     W/2, W/2;

        // _point_holder.reset(new ManyShiftsPointHolder(
        //     _pts,
        //     _g,
        //     W,
        //     _max_bound,
        //     shifts,
        //     0.5
        // ));

        _point_holder.reset(new UniformPointHolder(
            _pts,
            _g
        ));

        // int W = int(_max_bound / 100);

        // std::vector<PointHolder*> holders{
        //     new ShiftedPointHolder(
        //         _pts,
        //         _g,
        //         W,
        //         _max_bound,
        //         0,
        //         0
        //     ),
        //     new ShiftedPointHolder(
        //         _pts,
        //         _g,
        //         W,
        //         _max_bound,
        //         W/2,
        //         W/2
        //     ),
        //     new DiagonalPointHolder(
        //         _pts,
        //         _g,
        //         W
        //     )
        // };

        // _point_holder.reset(
        //     new MixturePointHolder(
        //         _pts,
        //         _g,
        //         holders
        //     )
        // );


        //
        // Calculate starting and target likelihoods
        //
        calculate_target_likelihood();
        calculate_current_histogram();
        calculate_starting_likelihood();

        //
        // Run!
        //
        cerr << "Running..." << endl;
        progress_bar = new boost::progress_display(_n_iters);
        auto period_time_begin = std::chrono::steady_clock::now();

        boost::filesystem::path stats_filename = boost::filesystem::path(_output_dir) / boost::filesystem::path("output." + to_string(_n_iters) + ".stats");
        ofstream stats_file;
        stats_file.open(stats_filename.string());
        
        long long int n_iter = 0;
        for (; n_iter < _n_iters; n_iter++) {

            (*progress_bar) += 1;   

            if ((n_iter % _output_every == 0)) {// && (n_iter > 0)
                const auto& output_points = _pts;   
                boost::filesystem::path output_filename = boost::filesystem::path(_output_dir) / boost::filesystem::path("output." + to_string(n_iter));
                saveArray(output_points.topRows(_print_only_first), output_filename.string(), _output_txt);
            }   

            _point_holder->draw_switch(two_old_contacts, two_new_contacts, two_chosen_inds);    

            // See if it improved
            log_ratio = compute_log_ratio(two_old_contacts, two_new_contacts);

            _point_holder->probability_of_switch(
                two_old_contacts, 
                two_new_contacts,
                probability_of_new_given_old,
                probability_of_old_given_new
            );

            accept_p = exp(log_ratio) * probability_of_old_given_new / probability_of_new_given_old;

            if (accept_p >= 1) {
                total_improved_delta++;
            }

            // If so, do it
            if (decide_to_accept(accept_p)) {
                // if (dynamic_cast<ShiftedPointHolder*>(_point_holder.get())->point_to_bin_number(two_old_contacts.row(0)) == 1) {
                //     cout << accept_p << " " << endl << two_old_contacts << endl << two_new_contacts << endl << endl;
                // }
                
                
                // Update histogram
                update_current_histogram(two_old_contacts, two_new_contacts);

                // Internal maintenance before actually performing switch
                _point_holder->perform_switch(two_new_contacts, two_chosen_inds);

                // Perform the switch
                perform_switch_in_points(two_new_contacts, two_chosen_inds);

                // Stats
                accept_count++;
                total_accept_count++;
            }

            
            if ((n_iter % _rate_every == 0) && (n_iter > 0)) {
                int max_abs_diff = calculate_max_abs_diff();
                // if (max_abs_diff < 5) {
                //     break;
                // }

                auto total_elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>
                    (std::chrono::steady_clock::now() - begin).count();               
                
                auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>
                    (std::chrono::steady_clock::now() - period_time_begin).count();                

                stats_file << boost::format("#its: %10i | 🕑: %6.1f [min] | it/s: %11.1f | Δ>1: %2.5f%% | Accept: %2.5f%% | max(abs(diff)): %8i | mean(abs(diff)): %6.2f | %%LL: %1.6f\n")
                    % n_iter
                    % (total_elapsed_seconds / 60)
                    % (_rate_every / elapsed_seconds)
                    % (float(total_improved_delta) / _rate_every * 100)
                    % (float(accept_count) / _rate_every * 100)
                    % max_abs_diff
                    % (float(_l1_distance) / _n_bins)
                    % ((calculate_current_likelihood() - _starting_log_likelihood) / (_target_log_likelihood - _starting_log_likelihood))
                    << flush;
                
                total_improved_delta = 0;
                accept_count = 0;
                accept_count_same_bin = 0;
                accept_count_general = 0;
                total_same_bin = 0;
                total_general = 0;
                period_time_begin = std::chrono::steady_clock::now();
            }           

            // Do whatever is needed to do after each step
            post_step(n_iter);
        }
        

        chrono::steady_clock::time_point end = chrono::steady_clock::now();
        cout << boost::format("Total acceptance rate: %8d / %10d = %1.4f\n")
            % total_accept_count
            % n_iter
            % (static_cast<double>(total_accept_count) / n_iter);
        cout << "Running time: " << chrono::duration_cast<chrono::seconds> (end - begin).count() << "[sec]" << endl;

        const auto& output_points = _pts;   
        boost::filesystem::path output_filename = boost::filesystem::path(_output_dir) / boost::filesystem::path("output." + to_string(_n_iters));
        saveArray(output_points, output_filename.string(), _output_txt); 

        return;
    }        
};