#pragma once
#include "common.h"

class PointHolder {    
  public:
    PointMatrix& _pts;
    mt19937& _g;
    int _n_points;
    std::uniform_int_distribution<> _uniform_point_sampler; 
    uniform_real_distribution<double> _unit_interval_sampler;

    PointHolder(PointMatrix& pts,
                mt19937& g) :
        _pts(pts),
        _g(g),
        _n_points(pts.rows()),
        _uniform_point_sampler(0, _n_points-1)
        { }

    virtual int draw_random_point() {
        return _uniform_point_sampler(_g);
    }

    virtual void draw_switch(
        TwoContactsMatrix& two_old_contacts,
        TwoContactsMatrix& two_new_contacts,
        pair<int, int>& two_chosen_inds,
        int first_chosen_index = -1) { }    

    virtual void probability_of_switch(
        TwoContactsMatrix& two_old_contacts, 
        TwoContactsMatrix& two_new_contacts,
        double& probability_of_new_given_old,
        double& probability_of_old_given_new) { }

    virtual void perform_switch(
        TwoContactsMatrix& two_new_contacts, 
        pair<int, int>& two_chosen_inds) { }

    virtual ~PointHolder() { }
};

class UniformPointHolder : public PointHolder {
  public:
    UniformPointHolder(
        PointMatrix& pts,
        mt19937& g) :
        PointHolder(pts, g) { }

    virtual void draw_switch(
        TwoContactsMatrix& two_old_contacts,
        TwoContactsMatrix& two_new_contacts,
        pair<int, int>& two_chosen_inds,
        int first_chosen_index = -1) {

        if (first_chosen_index != -1) {
            two_chosen_inds.first = first_chosen_index;
        } else {
            two_chosen_inds.first = draw_random_point();
        }
        two_chosen_inds.second = draw_random_point();
                         
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

    virtual void probability_of_switch(
        TwoContactsMatrix& two_old_contacts, 
        TwoContactsMatrix& two_new_contacts,
        double& probability_of_new_given_old,
        double& probability_of_old_given_new) {
        // All probabilities are uniform
        probability_of_new_given_old = 1.0 / (_n_points * (_n_points - 1));
        probability_of_old_given_new = 1.0 / (_n_points * (_n_points - 1));
    }   

    virtual void perform_switch(
        TwoContactsMatrix& two_new_contacts, 
        pair<int, int>& two_chosen_inds) {
        // No actual internal maintainance to do
    } 

    virtual ~UniformPointHolder() {}
};

class ShiftedPointHolder : public PointHolder {
  public:
    
    int _W;
    int _max_bound;
    int _shift0;
    int _shift1;

    int _n_bins;
    std::vector<int> _point_to_bin;
    std::vector<int> _point_to_index_within_bin;    
    boost::ptr_vector<std::vector<int>> _bin_to_contacts;

    ShiftedPointHolder(PointMatrix& pts,
                mt19937& g,
                int W,
                int max_bound,
                int shift0,
                int shift1) :
        PointHolder(
            pts,
            g
        ),
        _W(W),        
        _max_bound(max_bound),
        _shift0(shift0),
        _shift1(shift1),
        
        _n_bins(int(ceil(max_bound / double(W)) + 1)),
        _point_to_bin(_n_points, -1),
        _point_to_index_within_bin(_n_points, -1)
    {
        // Initialize
        for (int i = 0; i < _n_bins * _n_bins; i++) {
            _bin_to_contacts.push_back(new std::vector<int>());  
        }

        // Where do the points fall            
        for (int i = 0; i < _n_points; i++) {
            // int bin_idx = int((_pts(i, 0) + _shift0) / _W) * _n_bins + int((_pts(i, 1) + _shift1) / _W);
            int bin_idx = point_to_bin_number(_pts.row(i));
            insert_point(bin_idx, i);       
        }
    }

    virtual ~ShiftedPointHolder() { }
       
    virtual void insert_point(int bin_number, int point_index) {
        if (_point_to_bin[point_index] == bin_number) { 
            return;
        }

        _point_to_bin[point_index] = bin_number;
        _point_to_index_within_bin[point_index] = _bin_to_contacts[bin_number].size();
        _bin_to_contacts[bin_number].push_back(point_index);        
    }  

    virtual void erase_point(int bin_number, int point_index) {
        if (_point_to_bin[point_index] != bin_number) { 
            return;
        }

        int current_index = _point_to_index_within_bin[point_index];
        int current_size = _bin_to_contacts[bin_number].size();
        if (current_index == current_size-1) {
            _bin_to_contacts[bin_number].pop_back();
        } else {
            _bin_to_contacts[bin_number].at(current_index) = _bin_to_contacts[bin_number].at(current_size-1);
            _bin_to_contacts[bin_number].pop_back();
            _point_to_index_within_bin[_bin_to_contacts[bin_number].at(current_index)] = current_index;
        }

        _point_to_bin[point_index] = -1;
        _point_to_index_within_bin[point_index] = -1;
    }

    virtual int point_to_bin_number(const ContactsVector& point) {
        return int((point(0) + _shift0) / _W) * _n_bins + int((point(1) + _shift1) / _W);
    }
   
    virtual int draw_random_point_from_bin(int bin_number) {
        return int(_unit_interval_sampler(_g) * _bin_to_contacts[bin_number].size());
    }        
    
    virtual void draw_switch(
        TwoContactsMatrix& two_old_contacts,
        TwoContactsMatrix& two_new_contacts,
        pair<int, int>& two_chosen_inds,
        int first_chosen_index = -1) {
        
        // Draw two points
        do {
            // Draw first point uniformly 
            // (i.e., proportionally from each bin according to the number of points in the bin)
            if (first_chosen_index != -1) {
                two_chosen_inds.first = first_chosen_index;
            } else {
                two_chosen_inds.first = draw_random_point();
            }

            // Draw the second point from the same bin
            two_chosen_inds.second = draw_random_point_from_bin(_point_to_bin[two_chosen_inds.first]);
        } while (two_chosen_inds.second == two_chosen_inds.first);
                        
        if (two_chosen_inds.second < two_chosen_inds.first) {
            std::swap(two_chosen_inds.first, two_chosen_inds.second);            
        }

        two_old_contacts(0, 0) = _pts(two_chosen_inds.first, 0);
        two_old_contacts(0, 1) = _pts(two_chosen_inds.first, 1);
        two_old_contacts(1, 0) = _pts(two_chosen_inds.second, 0);
        two_old_contacts(1, 1) = _pts(two_chosen_inds.second, 1);

        // Flip a coin to decide what switch to see
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

    virtual void probability_of_switch(
        TwoContactsMatrix& two_old_contacts, 
        TwoContactsMatrix& two_new_contacts,
        double& probability_of_new_given_old,
        double& probability_of_old_given_new) {
        // Shift
        // Matrix2i shifted_old_contacts, shifted_new_contacts;
        // shifted_old_contacts(0, 0) = two_old_contacts(0, 0) + _shift0;
        // shifted_old_contacts(0, 1) = two_old_contacts(0, 1) + _shift1;
        // shifted_old_contacts(1, 0) = two_old_contacts(1, 0) + _shift0;
        // shifted_old_contacts(1, 1) = two_old_contacts(1, 1) + _shift1;

        // shifted_new_contacts(0, 0) = two_new_contacts(0, 0) + _shift0;
        // shifted_new_contacts(0, 1) = two_new_contacts(0, 1) + _shift1;
        // shifted_new_contacts(1, 0) = two_new_contacts(1, 0) + _shift0;
        // shifted_new_contacts(1, 1) = two_new_contacts(1, 1) + _shift1;

        Vector2i old_ind1d;
        // old_ind1d(0) = int(shifted_old_contacts(0, 0) / _W) * _n_bins + int(shifted_old_contacts(0, 1) / _W);
        // old_ind1d(1) = int(shifted_old_contacts(1, 0) / _W) * _n_bins + int(shifted_old_contacts(1, 1) / _W);
        old_ind1d(0) = point_to_bin_number(two_old_contacts.row(0));
        old_ind1d(1) = point_to_bin_number(two_old_contacts.row(1));

        Vector2i new_ind1d;
        // new_ind1d(0) = int(shifted_new_contacts(0, 0) / _W) * _n_bins + int(shifted_new_contacts(0, 1) / _W);
        // new_ind1d(1) = int(shifted_new_contacts(1, 0) / _W) * _n_bins + int(shifted_new_contacts(1, 1) / _W);
        new_ind1d(0) = point_to_bin_number(two_new_contacts.row(0));
        new_ind1d(1) = point_to_bin_number(two_new_contacts.row(1));

        // If they are not in the same bin, the prob is 0
        if (old_ind1d(0) != old_ind1d(1)) {
            probability_of_new_given_old = 0.0;
        } else {
            // The probablity to get this particular switch is:
            // - The probability to draw the first point (1 / # of points)
            // - The probability to draw the second point given the first (1 / (# points in bin - 1))
            // - The probability to draw that particular switch given the two points (1/2)
            probability_of_new_given_old = 1.0 / (_n_points * (_bin_to_contacts[old_ind1d(0)].size() - 1) * 2);
        }

        // The other side
        if (new_ind1d(0) != new_ind1d(1)) {
            probability_of_old_given_new = 0.0;
        } else {
            // If they are both in the same bin after the same switch, it could be shown (?) that 
            // It's the same bin as before, so the number of points in the bin doesn't change and 
            // therefore the probability doesn't change
            probability_of_old_given_new = 1.0 / (_n_points * (_bin_to_contacts[new_ind1d(0)].size() - 1) * 2);
        }
    }        

    virtual void perform_switch(TwoContactsMatrix& two_new_contacts, pair<int, int>& two_chosen_inds) {
        erase_point(_point_to_bin[two_chosen_inds.first], two_chosen_inds.first);
        erase_point(_point_to_bin[two_chosen_inds.second], two_chosen_inds.second);

        insert_point(point_to_bin_number(two_new_contacts.row(0)), two_chosen_inds.first);
        insert_point(point_to_bin_number(two_new_contacts.row(1)), two_chosen_inds.second); 
    }
};

/*
class ShfManyShifts {
  public:
    boost::ptr_vector<ShfPtholder> _holders;
    default_random_engine _rd;
    mt19937 _g;
    std::uniform_int_distribution<> _uniform_sampler;    
    PointMatrix& _pts;  
   
    ShfManyShifts(PointMatrix& pts,
                int W,
                int max_bound,
                int seed,
                MatrixXi shifts) :
        _rd(seed),
        _g(_rd()),
        _uniform_sampler(0, shifts.rows()-1),
        _pts(pts)
    {
        // cout << __LINE__ << endl;
        cout << "dims in shfmanyshufts constructor:" << " " << pts.rows() << " " << pts.cols() << endl;
        for (int i = 0; i < shifts.rows(); i++) {
            _holders.push_back(new ShfPtholder(pts, W, max_bound, 0, shifts(i, 0), shifts(i, 1)));
            //cerr << _holders[i]._sum_num_of_switches << endl;
        }        
        // printf("Address of x is %p\n", (void *)ip);  
        cout << "Address of holders._pts is" << " " << &_pts << endl;
        cout << "dims in shfmanyshufts member:" << " " << _pts.rows() << " " << _pts.cols() << endl;
        // cout << _pts(0,0) << endl;
        // cout << __LINE__ << endl;
        for (unsigned int i=0; i<_holders.size(); i++) {
            cout << "Address of holder[i]._pts is" << " " << &(_holders[i]._pts) << endl;
            // cout << _holders[i]._pts(0,0) << endl;
            cout << "dims in shf[i] member:" << " " << _holders[i]._pts.rows() << " " << _holders[i]._pts.cols() << endl;
            // cout << __LINE__ << endl;
        }
    }


    void draw_switch(TwoContactsMatrix& two_old_contacts, TwoContactsMatrix& two_new_contacts, pair<int, int>& two_chosen_inds) {
        // cout << __LINE__ << endl;
        int i =  _uniform_sampler(_g);
        // cout << __LINE__ << endl;
        // cout << i << endl;
        // cout << _holders.size() << endl;
        _holders[i].draw_switch(two_old_contacts, two_new_contacts, two_chosen_inds);
        // cout << __LINE__ << endl;
    }

    double probability_of_switch(TwoContactsMatrix& two_old_contacts, TwoContactsMatrix& two_new_contacts) {
        double ps = 0.0;
        for (unsigned int i = 0; i < _holders.size(); i++) {
            ps += _holders[i].probability_of_switch(two_old_contacts, two_new_contacts);
        }
        return (ps / _holders.size());
    }

    void perform_switch(TwoContactsMatrix& two_new_contacts, pair<int, int>& two_chosen_inds) {
        // cout << __LINE__ << endl;
        for (unsigned int i = 0; i < _holders.size(); i++) {
            _holders[i].perform_switch(two_new_contacts, two_chosen_inds);
        }
        _pts.row(two_chosen_inds.first) = two_new_contacts.row(0);
        _pts.row(two_chosen_inds.second) = two_new_contacts.row(1);
    }
};
*/