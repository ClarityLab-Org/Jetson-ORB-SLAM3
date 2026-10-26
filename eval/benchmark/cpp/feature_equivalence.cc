// Feature-level GPU-vs-CPU equivalence check (paper arXiv:2608.17874, Sec. 3.1.5).
// plan/paper_runbook.md, Part R4 ("features").
//
// Runs the repository's own ORB extractor twice on the same images -- once with the GPU front
// end (use_gpu = true) and once with the reference CPU extractor (use_gpu = false) -- and reports:
//   * total keypoint counts of each,
//   * the share of CPU keypoints that the GPU reproduces exactly (same x, y, octave),
//   * for those pairs, the mean Hamming distance of the 256-bit descriptors, the bit agreement,
//     and the share of descriptors that are bit-identical.
// Only the public ORB_SLAM3::ORBextractor API is used; the library itself is not modified.
//
// usage: feature_equivalence IMAGE_DIR OUT.json [--frames 205] [--every 10]
//                            [--nfeatures 1200] [--scale 1.2] [--levels 8] [--ini 20] [--min 7]
// Defaults are the EuRoC settings (Examples/Stereo-Inertial/EuRoC.yaml) and 205 frames, as in the paper.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <tuple>
#include <vector>

#include <dirent.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "ORBextractor.h"

using namespace std;

static vector<string> list_png(const string& dir)
{
    vector<string> out;
    DIR* d = opendir(dir.c_str());
    if (!d)
        return out;
    while (dirent* e = readdir(d))
    {
        string n = e->d_name;
        if (n.size() > 4 && n.substr(n.size() - 4) == ".png")
            out.push_back(dir + "/" + n);
    }
    closedir(d);
    sort(out.begin(), out.end());
    return out;
}

static int hamming(const uint8_t* a, const uint8_t* b)
{
    int d = 0;
    for (int i = 0; i < 32; ++i)
        d += __builtin_popcount(static_cast<unsigned>(a[i] ^ b[i]));
    return d;
}

// Key for "same position, scale and octave": coordinates rounded to 1/1000 px (i.e. exact up to
// float printing noise) plus the octave (the octave fixes the scale).
static tuple<long, long, int> key(const cv::KeyPoint& k)
{
    return make_tuple(lround(k.pt.x * 1000.0), lround(k.pt.y * 1000.0), k.octave);
}

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        cerr << "usage: feature_equivalence IMAGE_DIR OUT.json [--frames N] [--every K] [--nfeatures N]"
                " [--scale S] [--levels L] [--ini T] [--min T]" << endl;
        return 1;
    }
    string dir = argv[1], out = argv[2];
    int frames = 205, every = 10, nfeat = 1200, levels = 8, ini = 20, mn = 7;
    float scale = 1.2f;
    for (int i = 3; i + 1 < argc; i += 2)
    {
        string a = argv[i];
        double v = atof(argv[i + 1]);
        if (a == "--frames") frames = (int)v;
        else if (a == "--every") every = (int)v;
        else if (a == "--nfeatures") nfeat = (int)v;
        else if (a == "--scale") scale = (float)v;
        else if (a == "--levels") levels = (int)v;
        else if (a == "--ini") ini = (int)v;
        else if (a == "--min") mn = (int)v;
        else { cerr << "unknown option " << a << endl; return 1; }
    }

    vector<string> imgs = list_png(dir);
    if (imgs.empty())
    {
        cerr << "no .png images in " << dir << endl;
        return 1;
    }
    cv::Mat first = cv::imread(imgs[0], cv::IMREAD_GRAYSCALE);
    if (first.empty())
    {
        cerr << "cannot read " << imgs[0] << endl;
        return 1;
    }

    // Same constructor the tracker uses (src/Tracking.cc): last two arguments are the image size.
    ORB_SLAM3::ORBextractor gpu(nfeat, scale, levels, ini, mn, true, first.rows, first.cols);
    ORB_SLAM3::ORBextractor cpu(nfeat, scale, levels, ini, mn, false, first.rows, first.cols);

    long kp_gpu = 0, kp_cpu = 0, matched = 0, identical = 0, ham_sum = 0;
    int used = 0;
    for (size_t i = 0; i < imgs.size() && used < frames; i += every)
    {
        cv::Mat im = cv::imread(imgs[i], cv::IMREAD_GRAYSCALE);
        if (im.empty())
            continue;
        vector<cv::KeyPoint> kg, kc;
        cv::Mat dg, dc;
        vector<int> lap = {0, 0};
        gpu(im, cv::Mat(), kg, dg, lap);
        cpu(im, cv::Mat(), kc, dc, lap);
        kp_gpu += (long)kg.size();
        kp_cpu += (long)kc.size();

        map<tuple<long, long, int>, int> idx;
        for (size_t j = 0; j < kg.size(); ++j)
            idx.emplace(key(kg[j]), (int)j);
        for (size_t j = 0; j < kc.size(); ++j)
        {
            auto it = idx.find(key(kc[j]));
            if (it == idx.end())
                continue;
            ++matched;
            int h = hamming(dc.ptr<uint8_t>((int)j), dg.ptr<uint8_t>(it->second));
            ham_sum += h;
            if (h == 0)
                ++identical;
        }
        ++used;
    }
    if (used == 0 || kp_cpu == 0)
    {
        cerr << "no frames processed" << endl;
        return 1;
    }
    double exact_pct = 100.0 * matched / kp_cpu;
    double mean_ham = matched ? (double)ham_sum / matched : NAN;
    double bit_pct = matched ? 100.0 * (1.0 - mean_ham / 256.0) : NAN;
    double ident_pct = matched ? 100.0 * identical / matched : NAN;

    ofstream f(out);
    f << "{\n"
      << "  \"frames\": " << used << ",\n"
      << "  \"every\": " << every << ",\n"
      << "  \"image_dir\": \"" << dir << "\",\n"
      << "  \"kp_gpu\": " << kp_gpu << ",\n"
      << "  \"kp_cpu\": " << kp_cpu << ",\n"
      << "  \"matched\": " << matched << ",\n"
      << "  \"exact_kp_pct\": " << exact_pct << ",\n"
      << "  \"mean_hamming_bits\": " << mean_ham << ",\n"
      << "  \"bit_agreement_pct\": " << bit_pct << ",\n"
      << "  \"identical_desc_pct\": " << ident_pct << "\n"
      << "}\n";
    cout << "frames " << used << "  keypoints GPU " << kp_gpu << " / CPU " << kp_cpu
         << "  exact " << exact_pct << " %  mean Hamming " << mean_ham << " bits  bit agreement "
         << bit_pct << " %  identical " << ident_pct << " %" << endl;
    return 0;
}
