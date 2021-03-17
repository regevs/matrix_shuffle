
#include <iostream>
#include <fstream>
#include <string>
#include <random>
#include <algorithm>
#include <iterator>
#include <thread>
#include <mutex>
#include <future>
#include <chrono>
#include <cmath>
#include <memory>
#include <unordered_set>
#include <stdint.h>

#include <assert.h>     /* assert */

#include <boost/progress.hpp>
#include <boost/program_options.hpp>
#include <boost/bind.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/ref.hpp>
#include <boost/unordered_map.hpp>
#include <boost/ptr_container/ptr_vector.hpp>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/format.hpp>

#include "hopscotch/hopscotch_map.h"

#include "Eigen/Dense"
#include "Eigen/Core"
#include "Eigen/Sparse"

using namespace std;
using namespace Eigen;
namespace po = boost::program_options;

mutex progress_bar_lock;
boost::progress_display* progress_bar;

typedef Matrix<double, Dynamic, 2, RowMajor> PointMatrix;
typedef Matrix2d TwoContactsMatrix;
typedef Vector2d ContactsVector;

const int VERY_SMALL_LOG = -100000;

const static Eigen::IOFormat CSVFormat(FullPrecision, DontAlignCols, ",", "\n");
const static Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ", "", "", " << ", ";");

bool saveArray(const PointMatrix& output_points, const std::string& file_path, bool as_txt)
{
    if (as_txt) {
        std::ofstream os(file_path.c_str());
        if (!os.is_open())
            return false;
        os << output_points.cast<int>() << endl;
        os.close();
    } else {
        size_t length = output_points.rows() * output_points.cols();
        std::ofstream os(file_path.c_str(), std::ios::binary | std::ios::out);
        if (!os.is_open())
            return false;
        os.write(reinterpret_cast<const char*>(output_points.data()), std::streamsize(length*sizeof(double)));
        os.close();
    }
    return true;
}


PointMatrix load_csv(const string& path, bool skip_header = false) {
    ifstream indata;
    indata.open(path);
    string line;
    vector<double> values;
    uint rows = 0;
    if (skip_header) {
        getline(indata, line);   
    }
    while (getline(indata, line)) {
        std::vector<std::string> strs;
        boost::split(strs, line, boost::is_any_of("\t "), boost::token_compress_on);

        for (auto& cell : strs) {
            if (cell.length() > 0) {
                values.push_back(stod(cell)); //** wouldn't stof be better
            }
        }
        ++rows;
    }

    return Map<const PointMatrix>(values.data(), rows, values.size()/rows);
}

void sort_filter_points(const PointMatrix& points, PointMatrix* out_points, int min_dist) {
    int n_out_row = 0;
    int temp; 
    for (int n_row = 0; n_row < points.rows(); n_row++) {
        int a = points(n_row, 0);
        int b = points(n_row, 1);
        if (a>b) {
            temp = a; a = b; b = temp;            
        }
        if ((b-a) >= min_dist) {
            (*out_points)(n_out_row, 0) = a;
            (*out_points)(n_out_row, 1) = b;
            n_out_row++;
        }
    }

    out_points->conservativeResize(n_out_row, NoChange_t::NoChange);
}


// https://www.gormanalysis.com/blog/random-numbers-in-cpp/
std::vector<int> sample_without_replacement(int k, int N, std::default_random_engine& gen){
    // Sample k elements from the range [1, N] without replacement
    // k should be <= N
    
    // Create an unordered set to store the samples
    std::unordered_set<int> samples;
    
    if (k == N) {
        for (int r = 0; r < N; r++) {
            samples.insert(r+1);
        }
    } else {
        // Sample and insert values into samples
        for (int r = N - k; r < N; ++r) {
            int v = std::uniform_int_distribution<>(1, r)(gen);
            if (!samples.insert(v).second) samples.insert(r);
        }
    }
    
    // Copy samples into vector
    std::vector<int> result(samples.begin(), samples.end());
    
    // Shuffle vector
    std::shuffle(result.begin(), result.end(), gen);
    
    return result;
}


void downsample_matrix(const PointMatrix& points, PointMatrix* out_points, int dsn) {

    std::random_device myRandomDevice;
    // Initialize a default_random_engine
    std::default_random_engine myRandomEngine(std::random_device{}());
    std::vector<int> chosen_indices = sample_without_replacement(dsn, points.rows(), myRandomEngine);
    cout << *std::max_element(chosen_indices.begin(), chosen_indices.end()) << endl;
    cout << *std::min_element(chosen_indices.begin(), chosen_indices.end()) << endl;
    for (int index = 0; index < dsn; index++) {
        out_points->row(index) = points.row(chosen_indices[index] - 1);  // -1 because the functions return from [1,N]
    }

}

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

    void draw_switch(TwoContactsMatrix& two_old_contacts,
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

    void perform_switch(TwoContactsMatrix& two_new_contacts,
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
                    draw_switch(two_old_contacts, two_new_contacts, two_chosen_inds, i);
                    if (((two_new_contacts(0, 1) - two_new_contacts(0, 0)) >= _min_dist) &&
                        ((two_new_contacts(1, 1) - two_new_contacts(1, 0)) >= _min_dist)) {
                        perform_switch(two_new_contacts, two_chosen_inds);
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
        double accept_p;

        long long int accept_count = 0;
        long long int accept_count_same_bin = 0;
        long long int total_same_bin = 0;
        long long int accept_count_general = 0;
        long long int total_general = 0;
        long long int total_accept_count = 0;
        long long int total_improved_delta = 0;

        PointMatrix out_pts(_pts.rows(), _pts.cols());

        //
        // Shuffle coordinates properly
        //
        if (_shuffle) {
            shuffle_coordinates();
            verify_minimal_distance();
        }

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

            draw_switch(two_old_contacts, two_new_contacts, two_chosen_inds);    

            // See if it improved
            log_ratio = compute_log_ratio(two_old_contacts, two_new_contacts);

            accept_p = exp(log_ratio);

            if (accept_p >= 1) {
                total_improved_delta++;
            }

            // If so, do it
            if (decide_to_accept(accept_p)) {
                // Update histogram
                update_current_histogram(two_old_contacts, two_new_contacts);

                // Perform the switch
                perform_switch(two_new_contacts, two_chosen_inds);

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

                stats_file << boost::format("#its: %10i | 🕑: %6.1f [min] | it/s: %11.1f | Δ>1 %%: %1.6f | Accept %%: %1.6f | max(abs(diff)): %8i | mean(abs(diff)): %6.2f | %%LL: %1.6f\n")
                    % n_iter
                    % (total_elapsed_seconds / 60)
                    % (_rate_every / elapsed_seconds)
                    % (float(total_improved_delta) / _rate_every)
                    % (float(accept_count) / _rate_every)
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
        double _sgd_regularization_penalty;

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
                double sgd_regularization_penalty
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
            _sgd_regularization_penalty(sgd_regularization_penalty),
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
                delta + _sgd_regularization_penalty * star_reg_delta + _sgd_regularization_penalty * smooth_reg_delta,
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

// ========================================================================
//
// MAIN
//
// ========================================================================


int main(int argc, char** argv) {
    Eigen::initParallel();

    // Print command line
    for (int i = 0; i < argc; ++i) {
        cout << argv[i] << ' ';
    }
    cout << endl;

    //
    // Parse flags
    //
    po::options_description desc("Allowed options");
    desc.add_options()
        ("points_filename", po::value<string>()->default_value("points.txt"), "Points filename")
        ("output_dir", po::value<string>()->default_value("output"), "Output directory")
        ("counts_filename", po::value<string>(), "RV table filename")
        ("dcounts_filename", po::value<string>(), "RV table filename")
        ("W", po::value<int>()->default_value(-1), "Window size, set -1 for all range (obsolete)") 
        ("seed", po::value<int>()->default_value(0), "Random seem")
        ("min_dist", po::value<int>()->default_value(1000), "Minimum distance in a contact to consider")
        ("n_iters", po::value<long long int>()->default_value(100), "Number of iterations")
        ("output_every", po::value<long long int>()->default_value(-1), "Output every")
        ("rate_every", po::value<long long int>()->default_value(1000), "Output acceptance rate every")
        ("output_txt", po::value<bool>()->default_value(false), "Output as plain txt or as binary (read with np.load)")
        ("print_only_first", po::value<int>()->default_value(-1), "For intermediate printouts, print only first points (-1 for all)")
        ("skip_header", po::value<bool>()->default_value(false), "Skip first line of points file")
        ("down_sample_f", po::value<float>()->default_value(1), "Fraction of contacts to downsample")
        ("shuffle", po::value<bool>()->default_value(true), "Shuffle points at first")
        ("mode", po::value<string>()->default_value("mh"), "Should be one of: greedy, mh, srf, sgd")
        ("srf_every", po::value<long long int>()->default_value(1000000), "SRF every")
        ("srf_n0", po::value<long long int>()->default_value(1000000), "n0")
        ("srf_exponent", po::value<double>()->default_value(0.5), "exponent")
        ("srf_multiplier", po::value<double>()->default_value(1), "multiplier")
        ("sgd_regularization_penalty", po::value<double>()->default_value(1), "sgd_regularization_penalty")
    ;
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);    

    // 
    // Load data
    //
    string points_filename = vm["points_filename"].as<string>();
    if (!boost::filesystem::exists(points_filename)) {
        cout << "No such file: " << points_filename << endl;
        return -1;
    }
    
    PointMatrix points = load_csv(points_filename, vm["skip_header"].as<bool>());
    
    cerr << "Loaded " << points.rows() << " points. First points:" << endl;
    cerr << points.topRows(5).cast<int>() << endl << endl;
    // saveArray(points, (boost::filesystem::path(vm["output_dir"].as<string>()) / boost::filesystem::path("tmptry.txt")).string(), true);
    // return(0);

    PointMatrix sorted_filtered_points;
    sorted_filtered_points.resize(points.rows(), points.cols());
    sort_filter_points(points, &sorted_filtered_points, vm["min_dist"].as<int>());
    cout << "After filtering for minimal distance, " << sorted_filtered_points.rows() << " points remain." << endl;

    PointMatrix sorted_filtered_ds_points;
    // if (vm["down_sample_f"].as<float>()!=-1 and vm["down_sample_f"].as<float>()!=1) {
    if (vm["down_sample_f"].as<float>()!=1) {
        // int dsn = std::min(static_cast<int>(points.rows()), vm["dsn"].as<int>());
        int dsn = static_cast<int>(static_cast<int>(points.rows())*vm["down_sample_f"].as<float>());
        cout << "dsn:" << dsn << endl;
        sorted_filtered_ds_points.resize(dsn, sorted_filtered_points.cols());
        downsample_matrix(sorted_filtered_points, &sorted_filtered_ds_points, dsn);
    } else {
        sorted_filtered_ds_points = sorted_filtered_points;
    }
    cerr << "Using " << sorted_filtered_ds_points.rows() << " downsampled points." << endl;

    string counts_filename = vm["counts_filename"].as<string>();
    if (!boost::filesystem::exists(counts_filename)) {
        cout << "No such file: " << counts_filename << endl;
        return -1;
    }    
    MatrixXd counts = load_csv(counts_filename, true);

    string dcounts_filename = vm["dcounts_filename"].as<string>();
    if (!boost::filesystem::exists(dcounts_filename)) {
        cout << "No such file: " << dcounts_filename << endl;
        return -1;
    }    
    MatrixXd dcounts = load_csv(dcounts_filename, true);

    string output_dir = vm["output_dir"].as<string>();
    if (!boost::filesystem::exists(output_dir)) {
        cout << "Creating directory " << output_dir << endl;
        boost::filesystem::create_directories(output_dir);
    }

    int min_bound = sorted_filtered_ds_points.minCoeff();
    int max_bound = sorted_filtered_ds_points.maxCoeff() + 1;
    cout << "Min and max of range: " << min_bound << " " << max_bound << endl;
    
    long long int n_iters = vm["n_iters"].as<long long int>();
    long long int output_every = vm["output_every"].as<long long int>();
    if (output_every == 0) {
        output_every = n_iters;
    } else if (output_every == -1) {
        output_every = n_iters/10;
    }

    int distance_bin_size = vm["W"].as<int>();
    if (distance_bin_size == -1) {
        distance_bin_size = max_bound;
    }
   
    if (vm["mode"].as<string>() == "mh") {
        MetropolisHastings GO(
            sorted_filtered_ds_points, 
            min_bound,
            max_bound,
            vm["min_dist"].as<int>(),
            vm["seed"].as<int>(),
            counts, 
            dcounts,
            n_iters, 
            output_dir, 
            output_every,
            vm["output_txt"].as<bool>(),
            // vm["output_debug"].as<string>(),
            vm["rate_every"].as<long long int>(),
            vm["print_only_first"].as<int>(),
            vm["shuffle"].as<bool>()
            );
        GO.run();
    } else if (vm["mode"].as<string>() == "greedy") {
        GreedyOptimizer GO(
            sorted_filtered_ds_points, 
            min_bound,
            max_bound,
            vm["min_dist"].as<int>(),
            vm["seed"].as<int>(),
            counts, 
            dcounts,
            n_iters, 
            output_dir, 
            output_every,
            vm["output_txt"].as<bool>(),
            // vm["output_debug"].as<string>(),
            vm["rate_every"].as<long long int>(),
            vm["print_only_first"].as<int>(),
            vm["shuffle"].as<bool>()
            );
        GO.run();
    } else if (vm["mode"].as<string>() == "srf") {
        StochasticRootFinder GO(
            sorted_filtered_ds_points, 
            min_bound,
            max_bound,
            vm["min_dist"].as<int>(),
            vm["seed"].as<int>(),
            counts, 
            dcounts,
            n_iters, 
            output_dir, 
            output_every,
            vm["output_txt"].as<bool>(),
            // vm["output_debug"].as<string>(),
            vm["rate_every"].as<long long int>(),
            vm["print_only_first"].as<int>(),
            vm["shuffle"].as<bool>(),
            vm["srf_every"].as<long long int>(),
            vm["srf_n0"].as<long long int>(),
            vm["srf_exponent"].as<double>(),
            vm["srf_multiplier"].as<double>()
            );
        GO.run();
    } else if (vm["mode"].as<string>() == "sgd") {
        StochasticOptimizer GO(
            sorted_filtered_ds_points, 
            min_bound,
            max_bound,
            vm["min_dist"].as<int>(),
            vm["seed"].as<int>(),
            counts, 
            dcounts,
            n_iters, 
            output_dir, 
            output_every,
            vm["output_txt"].as<bool>(),
            // vm["output_debug"].as<string>(),
            vm["rate_every"].as<long long int>(),
            vm["print_only_first"].as<int>(),
            vm["shuffle"].as<bool>(),
            vm["srf_every"].as<long long int>(),
            vm["srf_n0"].as<long long int>(),
            vm["srf_exponent"].as<double>(),
            vm["srf_multiplier"].as<double>(),
            vm["sgd_regularization_penalty"].as<double>()
            );
        GO.run();
    } else {
        cerr << "Unknown mode!" << endl;
        exit(-1);
    }
    
    return 0;

}
