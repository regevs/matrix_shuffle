#include "common.h"

// Various solvers
#include "solver.h"
#include "greedy_optimizer.h"
#include "stochastic_root_finder.h"
#include "stochastic_optimizer.h"
#include "mh_solver.h"

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
        ("points_filename", po::value<string>(), "Points filename")
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
