/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/

// Rover driver: a copy of stereo_realsense_D435i.cc for runs on the rover (plan/stage1.md, 4.4).
// It uses the ORB-SLAM3 library unchanged, through the public System API only. Differences from
// the original driver:
//  - 'q' or Ctrl-C ends the run cleanly (the original stops only via the viewer's Stop button)
//  - frame and keyframe trajectories are saved in TUM format after shutdown
//  - tracking state, tracked-point counts, camera position and tracking time go to events.csv
//  - --out DIR and --no-viewer options
//  - camera images are copied inside the librealsense callback, and the image size comes from
//    the frame itself (the original kept pointers into librealsense buffers and read a size that
//    was set only after the stream had started)

#include <signal.h>
#include <stdlib.h>
#include <unistd.h>
#include <termios.h>
#include <poll.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <cerrno>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <ctime>
#include <sstream>
#include <string>
#include <vector>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <librealsense2/rs.hpp>

#include <System.h>

using namespace std;

static volatile sig_atomic_t b_continue_session = 1;

static void exit_loop_handler(int)
{
    b_continue_session = 0;
}

static std::atomic<bool> g_quit{false};
static std::atomic<char> g_key{0};      // last key other than 'q'; used by the Stage 2 add-on

// Single keys from the terminal without Enter. ISIG stays on, so Ctrl-C keeps working.
static void keyboard_thread()
{
    termios old_t{}, cb{};
    if (!isatty(STDIN_FILENO) || tcgetattr(STDIN_FILENO, &old_t) != 0)
        return;
    cb = old_t;
    cb.c_lflag &= ~(ICANON | ECHO);
    cb.c_cc[VMIN] = 1;
    cb.c_cc[VTIME] = 0;
    tcsetattr(STDIN_FILENO, TCSANOW, &cb);
    while (!g_quit && b_continue_session)
    {
        pollfd p{STDIN_FILENO, POLLIN, 0};
        if (poll(&p, 1, 100) > 0 && (p.revents & POLLIN))
        {
            char c;
            if (read(STDIN_FILENO, &c, 1) == 1)
            {
                if (c == 'q')
                    g_quit = true;
                else
                    g_key = c;
            }
        }
    }
    tcsetattr(STDIN_FILENO, TCSANOW, &old_t);
}

class EventLog
{
public:
    explicit EventLog(const string& path) : f_(path)
    {
        f_ << "time,event,detail\n";
        f_.flush();
    }
    bool ok() const { return f_.good(); }
    void log(double t, const string& event, const string& detail)
    {
        f_ << fixed << setprecision(6) << t << ',' << event << ',' << detail << '\n';
        f_.flush();
    }
private:
    ofstream f_;
};

static bool make_dirs(const string& path)
{
    string cur;
    for (size_t i = 0; i < path.size(); ++i)
    {
        cur += path[i];
        if ((path[i] == '/' && i > 0) || i + 1 == path.size())
            if (mkdir(cur.c_str(), 0755) != 0 && errno != EEXIST)
                return false;
    }
    return true;
}

static string now_stamp()
{
    time_t t = time(nullptr);
    char b[32];
    strftime(b, sizeof(b), "%Y%m%d-%H%M%S", localtime(&t));
    return b;
}

static void usage()
{
    cerr << endl
         << "Usage: ./stereo_realsense_D435_rover path_to_vocabulary path_to_settings [--out DIR] [--no-viewer]"
         << endl;
}

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        usage();
        return 1;
    }

    string out;
    bool use_viewer = true;
    for (int i = 3; i < argc; ++i)
    {
        string a = argv[i];
        if (a == "--out" && i + 1 < argc)
            out = argv[++i];
        else if (a == "--no-viewer")
            use_viewer = false;
        else
        {
            cerr << "Unknown argument: " << a << endl;
            usage();
            return 1;
        }
    }
    if (out.empty())
    {
        const char* home = getenv("HOME");
        out = string(home ? home : ".") + "/runs/" + now_stamp();
    }
    while (out.size() > 1 && out.back() == '/')
        out.pop_back();
    if (!make_dirs(out))
    {
        cerr << "Cannot create output folder " << out << endl;
        return 1;
    }
    const string run_name = out.substr(out.find_last_of('/') + 1);

    EventLog ev(out + "/events.csv");
    if (!ev.ok())
    {
        cerr << "Cannot write " << out << "/events.csv" << endl;
        return 1;
    }

    struct sigaction sigIntHandler;
    sigIntHandler.sa_handler = exit_loop_handler;
    sigemptyset(&sigIntHandler.sa_mask);
    sigIntHandler.sa_flags = 0;
    sigaction(SIGINT, &sigIntHandler, NULL);
    // If the console logger (tee) goes away, keep running and still save, instead of dying on SIGPIPE.
    signal(SIGPIPE, SIG_IGN);

    rs2::context ctx;
    rs2::device_list devices = ctx.query_devices();
    if (devices.size() == 0)
    {
        std::cerr << "No device connected, please connect a RealSense device" << std::endl;
        return 1;
    }
    rs2::device selected_device = devices[0];
    const string cam_name = selected_device.supports(RS2_CAMERA_INFO_NAME) ? selected_device.get_info(RS2_CAMERA_INFO_NAME) : "?";
    const string cam_serial = selected_device.supports(RS2_CAMERA_INFO_SERIAL_NUMBER) ? selected_device.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER) : "?";
    const string cam_fw = selected_device.supports(RS2_CAMERA_INFO_FIRMWARE_VERSION) ? selected_device.get_info(RS2_CAMERA_INFO_FIRMWARE_VERSION) : "?";
    const string cam_usb = selected_device.supports(RS2_CAMERA_INFO_USB_TYPE_DESCRIPTOR) ? selected_device.get_info(RS2_CAMERA_INFO_USB_TYPE_DESCRIPTOR) : "?";
    cout << "[ROVER] camera: " << cam_name << "  serial " << cam_serial << "  firmware " << cam_fw << "  USB " << cam_usb << endl;
    if (!cam_usb.empty() && cam_usb[0] != '3')
        cout << "[ROVER] WARNING: camera is not on USB 3 -- check cable and port" << endl;

    // Same sensor settings as the original driver: auto-exposure capped at 5 ms, projector off.
    for (rs2::sensor sensor : selected_device.query_sensors())
    {
        const string name = sensor.supports(RS2_CAMERA_INFO_NAME) ? sensor.get_info(RS2_CAMERA_INFO_NAME) : "";
        if (name != "Stereo Module")
            continue;
        if (sensor.supports(RS2_OPTION_ENABLE_AUTO_EXPOSURE))
            sensor.set_option(RS2_OPTION_ENABLE_AUTO_EXPOSURE, 1);
        if (sensor.supports(RS2_OPTION_AUTO_EXPOSURE_LIMIT))
            sensor.set_option(RS2_OPTION_AUTO_EXPOSURE_LIMIT, 5000);
        if (sensor.supports(RS2_OPTION_EMITTER_ENABLED))
            sensor.set_option(RS2_OPTION_EMITTER_ENABLED, 0);
    }

    rs2::pipeline pipe;
    rs2::config cfg;
    cfg.enable_device(cam_serial);
    cfg.enable_stream(RS2_STREAM_INFRARED, 1, 640, 480, RS2_FORMAT_Y8, 30);
    cfg.enable_stream(RS2_STREAM_INFRARED, 2, 640, 480, RS2_FORMAT_Y8, 30);

    std::mutex frame_mutex;
    std::condition_variable frame_cv;
    cv::Mat imL_buf, imR_buf;
    double ts_buf = -1.0;
    bool frame_ready = false;
    int frames_since_read = 0;

    auto frame_callback = [&](const rs2::frame& frame)
    {
        rs2::frameset fs = frame.as<rs2::frameset>();
        if (!fs)
            return;
        rs2::video_frame l = fs.get_infrared_frame(1);
        rs2::video_frame r = fs.get_infrared_frame(2);
        if (!l || !r)
            return;
        const double t = fs.get_timestamp() * 1e-3;
        cv::Mat L(cv::Size(l.get_width(), l.get_height()), CV_8U, (void*)l.get_data(), l.get_stride_in_bytes());
        cv::Mat R(cv::Size(r.get_width(), r.get_height()), CV_8U, (void*)r.get_data(), r.get_stride_in_bytes());
        std::lock_guard<std::mutex> lk(frame_mutex);
        if (std::abs(ts_buf - t) < 0.001)       // the same frameset delivered twice
            return;
        L.copyTo(imL_buf);
        R.copyTo(imR_buf);
        ts_buf = t;
        frame_ready = true;
        ++frames_since_read;
        frame_cv.notify_one();
    };

    rs2::pipeline_profile pipe_profile = pipe.start(cfg, frame_callback);

    rs2::stream_profile cam_left = pipe_profile.get_stream(RS2_STREAM_INFRARED, 1);
    rs2::stream_profile cam_right = pipe_profile.get_stream(RS2_STREAM_INFRARED, 2);
    const rs2_intrinsics intr = cam_left.as<rs2::video_stream_profile>().get_intrinsics();
    const rs2_extrinsics ext = cam_right.get_extrinsics_to(cam_left);
    {
        ostringstream s;    // own stream, so cout's number format stays as ORB-SLAM3 expects
        s << fixed << setprecision(4)
          << "[ROVER] Infrared 1 intrinsics: fx=" << intr.fx << " fy=" << intr.fy
          << " cx=" << intr.ppx << " cy=" << intr.ppy << "  (" << intr.width << "x" << intr.height << ")\n"
          << "[ROVER] stereo baseline |tx| = " << std::fabs(ext.translation[0]) << " m\n";
        cout << s.str() << flush;
    }

    {
        ofstream info(out + "/run_info.txt");
        info << "run: " << run_name << "\n"
             << "date: " << now_stamp() << "\n"
             << "command:";
        for (int i = 0; i < argc; ++i)
            info << ' ' << argv[i];
        info << "\n"
             << "settings: " << argv[2] << "\n"
             << "viewer: " << (use_viewer ? "on" : "off") << "\n"
             << "CPU_ORB: " << (getenv("CPU_ORB") ? "set (reference CPU extractor)" : "not set (GPU extractor)") << "\n"
             << "camera: " << cam_name << " serial " << cam_serial << " firmware " << cam_fw << " USB " << cam_usb << "\n"
             << fixed << setprecision(4)
             << "intrinsics_ir1: fx=" << intr.fx << " fy=" << intr.fy << " cx=" << intr.ppx << " cy=" << intr.ppy << "\n"
             << "baseline_m: " << std::fabs(ext.translation[0]) << "\n";
    }

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM3::System SLAM(argv[1], argv[2], ORB_SLAM3::System::STEREO, use_viewer, 0, run_name);
    const float imageScale = SLAM.GetImageScale();

    // Frames that arrived while the vocabulary was loading are not tracking drops; start counting now.
    {
        std::lock_guard<std::mutex> lk(frame_mutex);
        frames_since_read = frame_ready ? 1 : 0;
    }

    std::thread kb(keyboard_thread);
    cout << "[ROVER] running. Press q (or Ctrl-C) to stop and save. Output: " << out << endl;

    double timestamp = 0.0;
    cv::Mat im, imRight;
    int last_state = -100;
    double last_stat_t = -1.0;
    double track_ms_sum = 0.0, track_ms_max = 0.0;
    int track_n = 0;
    long frames_total = 0, dropped_total = 0;
    auto last_frame_wall = std::chrono::steady_clock::now();
    bool warned_no_frames = false;

    while (b_continue_session && !g_quit && !SLAM.isShutDown())
    {
        {
            std::unique_lock<std::mutex> lk(frame_mutex);
            if (!frame_cv.wait_for(lk, std::chrono::seconds(1), [&] { return frame_ready; }))
            {
                if (!warned_no_frames && std::chrono::steady_clock::now() - last_frame_wall > std::chrono::seconds(2))
                {
                    cout << "[ROVER] WARNING: no camera frames for 2 s" << endl;
                    warned_no_frames = true;
                }
                continue;       // re-check the exit conditions
            }
            if (frames_since_read > 1)
            {
                cout << frames_since_read - 1 << " dropped frs\n";
                dropped_total += frames_since_read - 1;
            }
            frames_since_read = 0;
            timestamp = ts_buf;
            im = imL_buf.clone();
            imRight = imR_buf.clone();
            frame_ready = false;
        }
        last_frame_wall = std::chrono::steady_clock::now();
        warned_no_frames = false;
        ++frames_total;

        if (imageScale != 1.f)
        {
            int width = im.cols * imageScale;
            int height = im.rows * imageScale;
            cv::resize(im, im, cv::Size(width, height));
            cv::resize(imRight, imRight, cv::Size(width, height));
        }

        const auto t0 = std::chrono::steady_clock::now();
        // Stereo images are already rectified.
        Sophus::SE3f Tcw = SLAM.TrackStereo(im, imRight, timestamp);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        track_ms_sum += ms;
        track_ms_max = std::max(track_ms_max, ms);
        ++track_n;

        const int state = SLAM.GetTrackingState();
        if (state != last_state)
        {
            ev.log(timestamp, "state", to_string(state));
            last_state = state;
        }
        if (SLAM.MapChanged())      // loop closure or merge in the current map (hint only)
            ev.log(timestamp, "map_changed", "");

        const vector<ORB_SLAM3::MapPoint*> mps = SLAM.GetTrackedMapPoints();
        const long n_tracked = std::count_if(mps.begin(), mps.end(), [](ORB_SLAM3::MapPoint* p) { return p != nullptr; });

        if (last_stat_t < 0.0 || timestamp - last_stat_t >= 1.0)
        {
            ev.log(timestamp, "tracked", to_string(n_tracked));
            ostringstream tm;
            tm << fixed << setprecision(1) << (track_n ? track_ms_sum / track_n : 0.0) << ' ' << track_ms_max;
            ev.log(timestamp, "track_ms", tm.str());
            track_ms_sum = 0.0;
            track_ms_max = 0.0;
            track_n = 0;
            if (state == 2)
            {
                const Eigen::Vector3f c = Tcw.inverse().translation();
                ostringstream ps;
                ps << fixed << setprecision(3) << c.x() << ' ' << c.y() << ' ' << c.z();
                ev.log(timestamp, "pos", ps.str());
            }
            last_stat_t = timestamp;
        }
    }

    cout << "[ROVER] stopping" << endl;
    ev.log(timestamp, "stop", g_quit ? "key" : (!b_continue_session ? "sigint" : "viewer"));
    pipe.stop();
    if (!SLAM.isShutDown())
        SLAM.Shutdown();
    // Shutdown() asks mapping and loop closing to finish but does not wait for them;
    // give a running loop correction time to complete before saving.
    cout << "[ROVER] waiting 5 s for mapping and loop closing to finish" << endl;
    std::this_thread::sleep_for(std::chrono::seconds(5));
    SLAM.SaveTrajectoryTUM(out + "/frames.txt");
    SLAM.SaveKeyFrameTrajectoryTUM(out + "/keyframes.txt");

    {
        ofstream info(out + "/run_info.txt", ios::app);
        info << "frames: " << frames_total << "\n"
             << "dropped_frames: " << dropped_total << "\n";
    }

    g_quit = true;
    if (kb.joinable())
        kb.join();
    cout << "[ROVER] saved to " << out << endl;
    cout << "System shutdown!\n";
    return 0;
}
