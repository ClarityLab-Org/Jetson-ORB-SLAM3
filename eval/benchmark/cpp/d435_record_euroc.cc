// Record the D435's infrared stereo pair in the EuRoC folder layout, so the repository's own,
// unmodified stereo_euroc program can replay it, under every configuration, on identical input.
// plan/paper_runbook.md, Part R6.
//
// Camera settings are the same as the repository's live driver
// (Examples/Stereo/stereo_realsense_D435i.cc): IR 1 + IR 2, 640x480 Y8 @ 30 FPS, projector off,
// auto-exposure capped at 5 ms.
//
// Output (DIR):
//   mav0/cam0/data/<t_ns>.png, mav0/cam1/data/<t_ns>.png   left / right IR images
//   mav0/cam0/data.csv, mav0/cam1/data.csv                  '#timestamp [ns],filename'
//   times.txt                                                one <t_ns> per line (stereo_euroc's times file)
//   recording_info.txt                                       camera, intrinsics, baseline, counts
//
// usage: d435_record_euroc DIR [--seconds N]        (stop early with q + Enter, or Ctrl-C)

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <poll.h>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <librealsense2/rs.hpp>

using namespace std;

static volatile sig_atomic_t g_stop = 0;
static void on_sigint(int) { g_stop = 1; }

static bool mkdirs(const string& p)
{
    string cur;
    for (size_t i = 0; i < p.size(); ++i)
    {
        cur += p[i];
        if ((p[i] == '/' && i > 0) || i + 1 == p.size())
            if (mkdir(cur.c_str(), 0755) != 0 && errno != EEXIST)
                return false;
    }
    return true;
}

struct Pair { int64_t t_ns; cv::Mat l, r; };

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        cerr << "usage: d435_record_euroc DIR [--seconds N]" << endl;
        return 1;
    }
    string out = argv[1];
    double seconds = 0;
    for (int i = 2; i + 1 < argc; i += 2)
        if (string(argv[i]) == "--seconds")
            seconds = atof(argv[i + 1]);
    while (out.size() > 1 && out.back() == '/')
        out.pop_back();
    if (!mkdirs(out + "/mav0/cam0/data") || !mkdirs(out + "/mav0/cam1/data"))
    {
        cerr << "cannot create " << out << endl;
        return 1;
    }
    signal(SIGINT, on_sigint);

    rs2::context ctx;
    rs2::device_list devs = ctx.query_devices();
    if (devs.size() == 0)
    {
        cerr << "No RealSense device connected" << endl;
        return 1;
    }
    rs2::device dev = devs[0];
    for (rs2::sensor s : dev.query_sensors())
    {
        string name = s.supports(RS2_CAMERA_INFO_NAME) ? s.get_info(RS2_CAMERA_INFO_NAME) : "";
        if (name != "Stereo Module")
            continue;
        if (s.supports(RS2_OPTION_ENABLE_AUTO_EXPOSURE)) s.set_option(RS2_OPTION_ENABLE_AUTO_EXPOSURE, 1);
        if (s.supports(RS2_OPTION_AUTO_EXPOSURE_LIMIT)) s.set_option(RS2_OPTION_AUTO_EXPOSURE_LIMIT, 5000);
        if (s.supports(RS2_OPTION_EMITTER_ENABLED)) s.set_option(RS2_OPTION_EMITTER_ENABLED, 0);
    }

    rs2::config cfg;
    cfg.enable_device(dev.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER));
    cfg.enable_stream(RS2_STREAM_INFRARED, 1, 640, 480, RS2_FORMAT_Y8, 30);
    cfg.enable_stream(RS2_STREAM_INFRARED, 2, 640, 480, RS2_FORMAT_Y8, 30);

    // Images are copied in the callback and written to disk by a separate thread, so slow PNG
    // writes never block the camera. A bounded queue: if the disk can't keep up, frames are
    // dropped and counted (and reported), never silently lost.
    mutex mtx;
    condition_variable cv_q;
    deque<Pair> q;
    const size_t kMaxQueue = 300;
    atomic<long> received{0}, dropped{0}, written{0};
    double last_ts = -1;

    auto cb = [&](const rs2::frame& fr) {
        rs2::frameset fs = fr.as<rs2::frameset>();
        if (!fs) return;
        rs2::video_frame l = fs.get_infrared_frame(1), r = fs.get_infrared_frame(2);
        if (!l || !r) return;
        double ts = fs.get_timestamp();                       // ms (host-domain "global time")
        if (std::fabs(ts - last_ts) < 0.001) return;          // the same frameset delivered twice
        last_ts = ts;
        Pair p;
        p.t_ns = llround(ts * 1e6);
        p.l = cv::Mat(cv::Size(l.get_width(), l.get_height()), CV_8U, (void*)l.get_data(), l.get_stride_in_bytes()).clone();
        p.r = cv::Mat(cv::Size(r.get_width(), r.get_height()), CV_8U, (void*)r.get_data(), r.get_stride_in_bytes()).clone();
        ++received;
        lock_guard<mutex> lk(mtx);
        if (q.size() >= kMaxQueue) { ++dropped; return; }
        q.push_back(std::move(p));
        cv_q.notify_one();
    };

    ofstream times(out + "/times.txt"), c0(out + "/mav0/cam0/data.csv"), c1(out + "/mav0/cam1/data.csv");
    c0 << "#timestamp [ns],filename\n";
    c1 << "#timestamp [ns],filename\n";
    atomic<bool> writing{true};
    thread writer([&] {
        vector<int> png = {cv::IMWRITE_PNG_COMPRESSION, 1};
        while (true)
        {
            Pair p;
            {
                unique_lock<mutex> lk(mtx);
                cv_q.wait_for(lk, chrono::milliseconds(100), [&] { return !q.empty(); });
                if (q.empty()) { if (!writing) break; continue; }
                p = std::move(q.front());
                q.pop_front();
            }
            string name = to_string(p.t_ns) + ".png";
            if (cv::imwrite(out + "/mav0/cam0/data/" + name, p.l, png) &&
                cv::imwrite(out + "/mav0/cam1/data/" + name, p.r, png))
            {
                times << p.t_ns << "\n";
                c0 << p.t_ns << "," << name << "\n";
                c1 << p.t_ns << "," << name << "\n";
                ++written;
            }
        }
    });

    rs2::pipeline pipe;
    rs2::pipeline_profile prof = pipe.start(cfg, cb);
    rs2_intrinsics in = prof.get_stream(RS2_STREAM_INFRARED, 1).as<rs2::video_stream_profile>().get_intrinsics();
    rs2_extrinsics ex = prof.get_stream(RS2_STREAM_INFRARED, 2).get_extrinsics_to(prof.get_stream(RS2_STREAM_INFRARED, 1));
    cout << "Recording to " << out << "  (q + Enter or Ctrl-C to stop"
         << (seconds > 0 ? ", or after " + to_string((int)seconds) + " s" : "") << ")" << endl;

    auto t0 = chrono::steady_clock::now();
    while (!g_stop)
    {
        if (seconds > 0 && chrono::duration<double>(chrono::steady_clock::now() - t0).count() >= seconds)
            break;
        pollfd pfd{STDIN_FILENO, POLLIN, 0};
        if (poll(&pfd, 1, 200) > 0 && (pfd.revents & POLLIN))
        {
            char c;
            if (read(STDIN_FILENO, &c, 1) == 1 && (c == 'q' || c == 'Q'))
                break;
        }
        static long last = -1;
        long w = written;
        if (w / 150 != last)
        {
            last = w / 150;
            size_t qs;
            { lock_guard<mutex> lk(mtx); qs = q.size(); }
            cout << "  written " << w << " pairs, queue " << qs << ", dropped " << dropped << endl;
        }
    }
    pipe.stop();
    writing = false;
    writer.join();
    double dur = chrono::duration<double>(chrono::steady_clock::now() - t0).count();

    ofstream info(out + "/recording_info.txt");
    info << fixed << setprecision(4)
         << "camera: " << dev.get_info(RS2_CAMERA_INFO_NAME) << " serial " << dev.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER)
         << " firmware " << dev.get_info(RS2_CAMERA_INFO_FIRMWARE_VERSION)
         << " usb " << (dev.supports(RS2_CAMERA_INFO_USB_TYPE_DESCRIPTOR) ? dev.get_info(RS2_CAMERA_INFO_USB_TYPE_DESCRIPTOR) : "?") << "\n"
         << "ir1_intrinsics: fx=" << in.fx << " fy=" << in.fy << " cx=" << in.ppx << " cy=" << in.ppy
         << " size=" << in.width << "x" << in.height << "\n"
         << "baseline_m: " << std::fabs(ex.translation[0]) << "\n"
         << "duration_s: " << dur << "\nreceived: " << received << "\nwritten: " << written
         << "\ndropped: " << dropped << "\n";
    cout << "done: " << written << " stereo pairs in " << dur << " s (" << dropped << " dropped)" << endl;
    if (dropped > 0)
        cout << "WARNING: frames were dropped (disk too slow?) -- the replay will see gaps" << endl;
    return 0;
}
