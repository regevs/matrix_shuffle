#pragma once

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

typedef Matrix<double, Dynamic, 2, RowMajor> PointMatrix;
typedef Matrix2d TwoContactsMatrix;
typedef Vector2d ContactsVector;

const int VERY_SMALL_LOG = -100000;
