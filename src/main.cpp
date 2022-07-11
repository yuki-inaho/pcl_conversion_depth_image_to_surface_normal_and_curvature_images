#include <thread>
#include <iostream>
#include <unordered_map>

#include <popl.hpp>
#include <yaml-cpp/yaml.h>

#include <pcl/features/integral_image_normal.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <chrono>
#include <opencv2/opencv.hpp>

#ifdef USE_BACKWARD_CPP
#include <backward.hpp>
#endif

float BAD_POINT_DEPTH_VALUE = std::numeric_limits<float>::quiet_NaN();

pcl::PointCloud<pcl::PointXYZ>::Ptr
ConvertDepthImageToPointClouds(const cv::Mat &depth_image_f, const std::unordered_map<std::string, double> &intrinsic_parameter_map)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    int image_width = depth_image_f.cols;
    int image_height = depth_image_f.rows;
    cloud->width = image_width;
    cloud->height = image_height;
    cloud->is_dense = false;
    cloud->resize(image_width * image_height);

    float fx = intrinsic_parameter_map.at("fx");
    float fy = intrinsic_parameter_map.at("fy");
    float cx = intrinsic_parameter_map.at("cx");
    float cy = intrinsic_parameter_map.at("cy");

    int32_t index = 0;
    for (int32_t v = 0; v < image_height; v++)
    {
        for (int32_t u = 0; u < image_width; index++, u++)
        {
            auto &point = (*cloud)[index];
            float depth_value = depth_image_f.at<float>(v, u);

            if (depth_value == 0)
            {
                point.x = BAD_POINT_DEPTH_VALUE;
                point.y = BAD_POINT_DEPTH_VALUE;
                point.z = BAD_POINT_DEPTH_VALUE;
            }
            else
            {
                point.x = (static_cast<float>(u) - cx) / fx * depth_value;
                point.y = (static_cast<float>(v) - cy) / fy * depth_value;
                point.z = depth_value;
            }
        }
    }
    return cloud;
}

void ConvertOrganizedSurfaceNormalsToCVImages(const pcl::PointCloud<pcl::Normal>::Ptr normals, cv::Mat &surface_normal_image, cv::Mat &surface_curvature_image)
{
    /* @TODO: add shape and type validation lines for output images
     */

    int image_width = normals->width;
    int image_height = normals->height;

    int32_t index = 0;
    for (int32_t v = 0; v < image_height; v++)
    {
        for (int32_t u = 0; u < image_width; index++, u++)
        {
            auto &point = (*normals)[index];
            if (std::isfinite(point.curvature))
            {
                surface_curvature_image.at<float>(v, u) = point.curvature;
                surface_normal_image.at<cv::Vec3f>(v, u)[0] = point.normal_x;
                surface_normal_image.at<cv::Vec3f>(v, u)[1] = point.normal_y;
                surface_normal_image.at<cv::Vec3f>(v, u)[2] = point.normal_z;
            }
        }
    }
}

pcl::PointCloud<pcl::Normal>::Ptr
EstimateNormalAndCurvature(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, const float &max_depth_change_factor, const float &normal_smoothing_size)
{
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    pcl::IntegralImageNormalEstimation<pcl::PointXYZ, pcl::Normal> normalEstimation;
    normalEstimation.setInputCloud(cloud);
    normalEstimation.setNormalEstimationMethod(normalEstimation.COVARIANCE_MATRIX);
    normalEstimation.setMaxDepthChangeFactor(max_depth_change_factor);
    normalEstimation.setNormalSmoothingSize(normal_smoothing_size);
    normalEstimation.compute(*normals);
    return normals;
}

cv::Mat ColorizeDepthImage(const cv::Mat &depth_image_f)
{
    cv::Mat depth_image_clamped, depth_image_uc8, depth_mask, depth_image_colorized;
    cv::inRange(depth_image_f, 0, 2.0, depth_mask);
    depth_image_f.copyTo(depth_image_clamped, depth_mask);
    depth_image_clamped.convertTo(depth_image_uc8, CV_8UC1, 255.0 / 2.0);
    cv::applyColorMap(depth_image_uc8, depth_image_colorized, cv::COLORMAP_JET);
    return depth_image_colorized;
}

cv::Mat ColorizeSurfaceNormalImage(const cv::Mat &surface_normal_image)
{
    cv::Mat surface_normal_image_colorized;
    cv::Mat surface_normal_image_biased = surface_normal_image + cv::Scalar(1.0f);
    surface_normal_image_biased.convertTo(surface_normal_image_colorized, CV_8UC3, 128);
    return surface_normal_image_colorized;
}

cv::Mat ColrizeCurvatureImage(const cv::Mat &surface_curvature_image)
{
    cv::Mat surface_curvature_image_uc8, surface_curvature_image_colorized;
    surface_curvature_image.convertTo(surface_curvature_image_uc8, CV_8UC1, 255);
    cv::applyColorMap(surface_curvature_image_uc8, surface_curvature_image_colorized, cv::COLORMAP_JET);
    return surface_curvature_image_colorized;
}

int main(int argc, char *argv[])
{
#ifdef USE_BACKWARD_CPP
    backward::SignalHandling sh;
#endif

    /* Setup argument parser
     */
    popl::OptionParser op("Allowed options");
    auto config_file_option = op.add<popl::Value<std::string>>("c", "config", "camera configuration file path", "../config/camera.yml");
    auto input_depth_image_option = op.add<popl::Value<std::string>>("i", "image", "target depth image path", "../data/freiburg1.png");
    auto max_depth_change_factor_option = op.add<popl::Value<float>>("m", "max_depth_change_factor", "max depth change factor for surface normal computation", 0.05f);
    auto normal_smoothing_size_option = op.add<popl::Value<float>>("n", "normal_smoothing_size", "normal smoothing size for surface normal computation", 10.0f);
    auto pcd_3d_visualizer_option = op.add<popl::Switch>("p", "enable_pcd_3d_visualizer", "if using the option, the pcd 3d visualizer will be launch");

    try
    {
        op.parse(argc, argv);
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        std::cerr << std::endl;
        std::cerr << op << std::endl;
        return EXIT_FAILURE;
    }

    /* Load camera configuration parameters
     */
    std::string config_file_path = config_file_option->value();
    YAML::Node yaml_parser;
    yaml_parser = YAML::LoadFile(config_file_path);
    std::unordered_map<std::string, double> intrinsic_parameter_map;

    std::vector<std::string> parameter_keys{"fx", "fy", "cx", "cy"};
    for (auto it = parameter_keys.begin(); it != parameter_keys.end(); ++it)
    {
        std::string key = *it;
        intrinsic_parameter_map[key] = yaml_parser["intrinsic"][key].as<double>();
    }
    int32_t image_width = yaml_parser["image_size"]["width"].as<int>();
    int32_t image_height = yaml_parser["image_size"]["height"].as<int>();

    /* Load the ta depth image
     */
    std::string input_depth_image_path = input_depth_image_option->value();
    cv::Mat depth_image = cv::imread(input_depth_image_path, cv::IMREAD_ANYDEPTH);
    if ((depth_image.cols != image_width) || (depth_image.rows != image_height))
    {
        std::cerr << "Image size of input does not correspond with configured values " << std::endl;
        return EXIT_FAILURE;
    }

    /* Scale pixel values of the input depth image to align its unit to [m]
     */
    cv::Mat depth_image_f;
    depth_image.convertTo(depth_image_f, CV_32FC1, 0.0001);

    /* Generate point cloud from the depth image
     */
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = ConvertDepthImageToPointClouds(depth_image_f, intrinsic_parameter_map);

    /* Surface normal calculation
     */

    float max_depth_change_factor = max_depth_change_factor_option->value();
    float normal_smoothing_size = normal_smoothing_size_option->value();

    std::chrono::system_clock::time_point start, end;
    start = std::chrono::system_clock::now();

    pcl::PointCloud<pcl::Normal>::Ptr normals = EstimateNormalAndCurvature(cloud, max_depth_change_factor, normal_smoothing_size);

    end = std::chrono::system_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Elapsed: " << elapsed << std::endl;

    /* Launch pcd viewer
     */
    bool launch_pcd_3d_visualizer = pcd_3d_visualizer_option->is_set();
    if (launch_pcd_3d_visualizer)
    {
        boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("Normals"));
        viewer->addPointCloud<pcl::PointXYZ>(cloud, "cloud");
        viewer->addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(cloud, normals, 20, 0.03, "normals");
        while (!viewer->wasStopped())
        {
            viewer->spinOnce(10);
            std::chrono::milliseconds duration(100); // [ms]
            std::this_thread::sleep_for(duration);
        }
    }

    /* Generate Surface Normal Image (quantized to 8UC3)
     */
    cv::Mat surface_normal_image = cv::Mat::zeros(cv::Size(image_width, image_height), CV_32FC3);
    cv::Mat surface_curvature_image = cv::Mat::zeros(cv::Size(image_width, image_height), CV_32FC1);
    ConvertOrganizedSurfaceNormalsToCVImages(normals, surface_normal_image, surface_curvature_image);

    /*  Draw 2D images
     */
    cv::Mat canvas;
    cv::Mat surface_normal_image_colorized = ColorizeSurfaceNormalImage(surface_normal_image);
    cv::Mat surface_curvature_image_colorized = ColrizeCurvatureImage(surface_curvature_image);
    cv::Mat depth_image_colorized = ColorizeDepthImage(depth_image_f);

    cv::hconcat(depth_image_colorized, surface_normal_image_colorized, canvas);
    cv::hconcat(canvas.clone(), surface_curvature_image_colorized, canvas);
    cv::imshow("depth & surface normal & curvature images", canvas);
    cv::waitKey(-1);

    return 0;
}
