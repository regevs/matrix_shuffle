#include "common.h"

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