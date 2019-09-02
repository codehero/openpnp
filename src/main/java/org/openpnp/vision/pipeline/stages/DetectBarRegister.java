package org.openpnp.vision.pipeline.stages;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;
import org.openpnp.vision.pipeline.CvPipeline;
import org.openpnp.vision.pipeline.CvStage;
import org.openpnp.vision.pipeline.Property;
import org.openpnp.vision.pipeline.Stage;
import org.simpleframework.xml.Attribute;
import org.pmw.tinylog.Logger;

/**
 * Finds lines in the working image and stores the results as a List<Circle> on the model. 
 */
@Stage(description="Finds circles in the working image. Diameter and spacing can be specified.")
public class DetectBarRegister extends CvStage {

    @Attribute
    @Property(description = "Distance resolution from center.")
    private double rho = 0.5;

    @Attribute
    @Property(description = "Angular resolution, in degrees.")
    private double theta = 0.1;

    @Attribute
    @Property(description = "Minimum accumulator count.")
    private int threshold = 1000;

    @Attribute
    @Property(description = "Minimum separation between bar sides, as percent of scene diagonal.")
    private double minSeparation= 10.0;

    @Attribute
    @Property(description = "Max angular divergence between sides of bar, in degrees.")
    private double maxAngularDivergence = 2.0;

    @Attribute
    @Property(description = "Max first pass angular divergence.")
    private double maxAngularFirstPass = 4.0;

    @Attribute
    @Property(description = "Angular threshold of bar parallelness.")
    private double barParallelness = 0.2;

    @Attribute
    @Property(description = "Side filter for reassessing edge pixels, percent of screen diagonal.")
    private double sideFilter = 1;


    public double getRho() {
        return this.rho;
    }

    public void setRho(double v) {
        this.rho = v;
    }

    public double getTheta() {
        return this.theta;
    }

    public void setTheta(double v) {
        this.theta = v;
    }

    public int getThreshold() {
        return this.threshold;
    }

    public void setThreshold(int v) {
        this.threshold = v;
    }

    public double getMinSeparation() {
        return this.minSeparation;
    }

    public void setMinSeparation(double v) {
        this.minSeparation = v;
    }

    public double getMaxAngularDivergence() {
        return this.maxAngularDivergence;
    }

    public void setMaxAngularDivergence(double v) {
        this.maxAngularDivergence = v;
    }

    public double getMaxAngularFirstPass() {
        return this.maxAngularFirstPass;
    }

    public void setMaxAngularFirstPass(double v) {
        this.maxAngularFirstPass= v;
    }

    public double getBarParallelness() {
        return this.barParallelness;
    }

    public void setBarParallelness(double v) {
        this.barParallelness = v;
    }

    public double getSideFilter() {
        return this.sideFilter;
    }

    public void setSideFilter(double v) {
        this.sideFilter = v;
    }

		private double edgeDist(Point p, double width, double height){
			double dx = width - p.x;
			double dy = height - p.y;

			if(dy > p.y){
				dy = p.y;
			}

			if(dx > p.x){
				dx = p.x;
			}

			return (dx < dy) ? dx : dy;
		}

		private MatOfPoint renderLine(double rho, double theta, double t0, double t1){
			double a = Math.cos(theta);
			double b = Math.sin(theta);
			double x0 = a*rho;
			double y0 = b*rho;
			double x1 = x0 + t0*(-b);
			double y1 = y0 + t0*(a);
			double x2 = x0 + t1*(-b);
			double y2 = y0 + t1*(a);

			List<Point> tmp = new ArrayList<>();
			tmp.add(new Point(x1, y1));
			tmp.add(new Point(x2, y2));

			MatOfPoint mp = new MatOfPoint();
			mp.fromList(tmp);
			return mp;
		}

		private Point calcLineLineIntersection(Point p0, Point d0, Point p1, Point d1){
			double dx = p1.x - p0.x;
			double dy = p1.y - p0.y;

			double cross = d0.x * d1.y - d0.y * d1.x;
			double t = (dx * d1.y - dy * d1.x) / cross;
			Point p = new Point(p0.x + d0.x * t, p0.y + d0.y * t);
			return p;
		}

		private Point calcIntersection(double rp, double tp, double ra, double ta){
			/* Got a good reading.
				 Work in cartesian space to determine intersection 
				 against sides of bar */
			double a = Math.cos(tp);
			double b = Math.sin(tp);

			/* (-b, a) is direction of perpendicular */
			double o1x = a * rp;
			double o1y = b * rp;
			double d1x = -b;
			double d1y = a;

			a = Math.cos(ta);
			b = Math.sin(ta);
			double o2x = a * ra;
			double o2y = b * ra;
			double d2x = -b;
			double d2y = a;

			double dx = o2x - o1x;
			double dy = o2y - o2y;

			double cross = d1x * d2y - d1y * d2x;
			double t = (dx * d2y - dy * d2x) / cross;
			Point p = new Point(o1x + d1x * t, o1y + d1y * t);
			return p;
		}

    @Override
    public Result process(CvPipeline pipeline) throws Exception {
        Mat mat = pipeline.getWorkingImage();

				double m = mat.width();
				double q = mat.height();

				/* m will be the diagonal length. */
				m = m * m;
				q = q * q;
				m = Math.sqrt(m + q);

        Mat output = new Mat();

				/* Convert degrees to radians.*/
				double p_mt = this.theta * Math.PI / 180.0;
				double p_mad = this.maxAngularDivergence * Math.PI / 180.0;
				double p_mafp = this.maxAngularFirstPass * Math.PI / 180.0;
				double p_bp = this.barParallelness * Math.PI / 180.0;

				double p_mr = this.rho * m / 100.0;
				double p_ms = this.minSeparation * m / 100.0;
				double p_sf = this.sideFilter * m / 100.0;

				/* This is our return value. */
        List<MatOfPoint> retval = new ArrayList<>();

				/* This first pass is a way to get the major axis of the bar
				   There are 2 expected results:
					 1) A clustering of all theta values around the bar's major axis
					 2) 2 clusterings of rho values, 1 for each side of the bar.

					 Note that the end of the bar is going to cause variations in the
					 theta values. As observed, this is about +/- 2 degrees.

					 If the variation is greater, something is wrong in the calibration
					 setup.
				   */
        Imgproc.HoughLines(mat, output, p_mr, p_mt, this.threshold);


				Logger.trace("Hough Lines " + output.rows());
				if(0 == output.rows()){
	        return new Result(null, retval);
				}

				/* This is more of a sanity check.
					 Verify that initial angular divergence is respected*/
				double min_angle = output.get(0, 0)[1];
				double max_angle = output.get(0, 0)[1];
        for (int i = 0; i < output.rows(); i++) {
					double thi = output.get(i, 0)[1];
					if(thi < min_angle){
						min_angle = thi;
					}
					if(thi > max_angle){
						max_angle = thi;
					}
				}

				if(max_angle - min_angle > p_mafp){
					for (int i = 0; i < output.rows(); i++) {
						double rhi = output.get(i, 0)[0];
						double thi = output.get(i, 0)[1];
						Logger.trace("L " + rhi + " " + (thi / Math.PI * 180.0));
						retval.add(renderLine(rhi, thi, 1000, -1000));
					}

					/* just do a list of lines. */
	        return new Result(null, retval);
				}

				/* Now we need to determine the bounding rho values.
				   The rho values should be clustered around 2 values.
				   There will be a large gap between the 2 clusters.
				   Find the element, that when ordered by rho, has the
				   greatest difference to the next value.

					 When grouped by rho, each point will also show
					 a clustering theta.
				 */
				double rho_0_avg = 0;
				double rho_1_avg = 0;
				double theta_0_avg = 0;
				double theta_1_avg = 0;

				double rho_0[] = null;
				double theta_0[] = null;
				double rho_1[] = null;
				double theta_1[] = null;
				{
					/* First sort all the rho values. */
					double tmpArray[] = new double[output.rows()];
					for (int i = 0; i < output.rows(); i++) {
						tmpArray[i] = output.get(i, 0)[0];
					 }

					/* Next find the greatest separation between rho. */
					double greatest_dist = 0;
					int greatest_i = 0;
					java.util.Arrays.sort(tmpArray);
					for (int i = 0; i < output.rows() - 1; i++) {
						double diff = tmpArray[i + 1] - tmpArray[i];
						if(diff > greatest_dist){
							greatest_dist = diff;
							greatest_i = i;
						}
					}

					Logger.trace("P2 " + greatest_dist + " " + greatest_i);

					if(0 == greatest_dist){
						for (int i = 0; i < output.rows(); i++) {
							double rhi = output.get(i, 0)[0];
							double thi = output.get(i, 0)[1];
							Logger.trace("L " + rhi + " " + (thi / Math.PI * 180.0));
							retval.add(renderLine(rhi, thi, 1000, -1000));
						}

						/* just do a list of lines. */
						return new Result(null, retval);

					}

					/* Now create the rho,theta subarrays for each line. */
					rho_0 = new double[greatest_i + 1];
					theta_0 = new double[greatest_i + 1];
					rho_1 = new double[output.rows() - greatest_i - 1];
					theta_1 = new double[output.rows() - greatest_i - 1];

					/* Get the values for line 0 */
					for (int i = 0; i <= greatest_i; i++) {
						rho_0[i] = tmpArray[i];
						rho_0_avg += tmpArray[i];
						for (int k = 0; k < output.rows(); k++) {
							if(output.get(k, 0)[0] == tmpArray[i]){
								double thk = output.get(k, 0)[1];
								theta_0[i] = thk;
								theta_0_avg += thk;
							}
						}
					}
					rho_0_avg /= greatest_i + 1;
					theta_0_avg /= greatest_i + 1;

					/* Get the values for line 1 */
					for (int i = greatest_i + 1; i < output.rows(); i++) {
						rho_1[i - greatest_i - 1] = tmpArray[i];
						rho_1_avg += tmpArray[i];
						for (int k = 0; k < output.rows(); k++) {
							if(output.get(k, 0)[0] == tmpArray[i]){
								double thk = output.get(k, 0)[1];
								theta_1[i - greatest_i - 1] = thk;
								theta_1_avg += thk;
							}
						}
					}
					rho_1_avg /= output.rows() - greatest_i - 1;
					theta_1_avg /= output.rows() - greatest_i - 1;
				}

				Logger.trace("L0 " + rho_0_avg + " " + (theta_0_avg / Math.PI * 180.0));
				Logger.trace("L1 " + rho_1_avg + " " + (theta_1_avg / Math.PI * 180.0));

				if(Math.abs(rho_1_avg - rho_0_avg) < p_ms){
					Logger.trace("Bad rho diff "
							+ Math.abs(rho_1_avg - rho_0_avg) + " vs " + p_ms);

					for (int i = 0; i < output.rows(); i++) {
						double rhi = output.get(i, 0)[0];
						double thi = output.get(i, 0)[1];
						Logger.trace("L " + rhi + " " + (thi / Math.PI * 180.0));
						retval.add(renderLine(rhi, thi, 1000, -1000));
					}

					return new Result(null, retval);
				}

				/* At this point, we can be reasonably sure that the sides of the
				 bar appear correctly in the image, but still have imprecise results.
				 The hough transform bins "smear" angle and rho measurements
				 The end of the bar also induces a bias in the angle measurements
				 on each side
				 (the points on the end will add counts and skew the average angle. */

				/* Next step is extract the pixels that are close to the bar lines.
				 Though the runtime could be reduced slightly by only iterating pixels
				 that are on the lines, I will just scan the whole image and threshold
				 on distance to the line.

				 At the same time, the pixel coordinates in the line space also should
				 show a discontinuity on the end of the bar.
				 We need this discontinuity to improve filtering and determine the
				 final x/y location of the end of the bar.
				 */
				double dx0 = Math.cos(theta_0_avg);
				double dy0 = Math.sin(theta_0_avg);

				double dx1 = Math.cos(theta_1_avg);
				double dy1 = Math.sin(theta_1_avg);

        List<Point> line0 = new ArrayList<>();
        List<Point> line1 = new ArrayList<>();

				for(int i = 0; i < mat.rows(); ++i){
					for(int k = 0; k < mat.cols(); ++k){
						double v = mat.get(i, k)[0];
						if(v > 0){

							double rp = dx0 * k + dy0 * i;

							if(Math.abs(rp - rho_0_avg) < p_sf){
								Point p = new Point(k, i);
								line0.add(p);
								continue;
							}

							rp = dx1 * k + dy1 * i;
							if(Math.abs(rp - rho_1_avg) < p_sf){
								Point p = new Point(k, i);
								line1.add(p);
							}
						}
					}
				}

				MatOfPoint mline0 = new MatOfPoint();
				mline0.fromList(line0);

				MatOfPoint mline1 = new MatOfPoint();
				mline1.fromList(line1);

				/* Use linear regressions on the filtered points to really lock in
				   on the sides of the bars. */
        Mat fline0 = new Mat();
        Imgproc.fitLine(mline0, fline0, Imgproc.CV_DIST_L2, 0, 0.01, 0.01);

        Mat fline1 = new Mat();
        Imgproc.fitLine(mline1, fline1, Imgproc.CV_DIST_L2, 0, 0.01, 0.01);

				{
					theta_0_avg = Math.asin(fline0.get(0,0)[0]);
					rho_0_avg = Math.cos(theta_0_avg) * fline0.get(2,0)[0] + Math.sin(theta_0_avg) * fline0.get(3,0)[0];

					theta_1_avg = Math.asin(fline1.get(0,0)[0]);
					rho_1_avg = Math.cos(theta_1_avg) * fline1.get(2,0)[0] + Math.sin(theta_1_avg) * fline1.get(3,0)[0];

					Logger.trace("L0 " + rho_0_avg + " " + (theta_0_avg / Math.PI * 180.0));
					Logger.trace("L1 " + rho_1_avg + " " + (theta_1_avg / Math.PI * 180.0));

					/* The lines must be separated by a minimum distance */
					if(Math.abs(rho_1_avg - rho_0_avg) < p_ms){
						Logger.trace("Bad rho diff "
								+ Math.abs(rho_1_avg - rho_0_avg) + " vs " + p_ms);

						MatOfPoint mp = new MatOfPoint();
						mp.fromList(line0);
						retval.add(mp);

						MatOfPoint mp2 = new MatOfPoint();
						mp2.fromList(line1);
						retval.add(mp2);

						mp.release();
						mp2.release();
						return new Result(null, retval);
					}

					/* The angles must be parallel within a tolerance
						 Exceeding this tolerance indicates a miscalibrated
						 camera. */
					if(Math.abs(theta_1_avg - theta_0_avg) > p_bp){
						Logger.trace("Bad theta diff " + ((theta_1_avg - theta_0_avg) / Math.PI * 180.0));

						MatOfPoint mp = new MatOfPoint();
						mp.fromList(line0);
						retval.add(mp);

						MatOfPoint mp2 = new MatOfPoint();
						mp2.fromList(line1);
						retval.add(mp2);
						mp.release();
						mp2.release();
						return new Result(null, retval);
					}

					/* At this point we can be confident we have the best calculation
					 of the lines defining the sides of the bar.
					 Now that we have the sides, we have to find the end of the bar,
					 as well as the direction toward the OTHER end of the bar.
					 
					 Project points onto the lines defining the sides.
					 */

					 /* FIXME: This step must be made more robust by eliminating any possible outliers
				   along the bar edge. In ideal environments, only points along the bar appear,
				   but a more robust algorithm would look for discontinuites in the t coords
				   and filter out those points that are not clearly part of the bar. */

					dx0 = Math.cos(theta_0_avg);
					dy0 = Math.sin(theta_0_avg);

					dx1 = Math.cos(theta_1_avg);
					dy1 = Math.sin(theta_1_avg);


					double t0_min = Double.POSITIVE_INFINITY;
					double t0_max = -Double.POSITIVE_INFINITY;
					double t0_edge_dist = Double.POSITIVE_INFINITY;
					for(int i = 0; i < line0.size(); ++i){
						Point p = line0.get(i);
						double rt = dx0 * p.y + - dy0 * p.x;

						if(rt > t0_max){
							t0_max = rt;
							double ed = edgeDist(p, mat.width(), mat.height());
							if(ed < Math.abs(t0_edge_dist)){
								t0_edge_dist = ed;
							}
						}
						if(rt < t0_min){
							t0_min = rt;

							double ed = edgeDist(p, mat.width(), mat.height());
							if(ed < Math.abs(t0_edge_dist)){
								t0_edge_dist = -ed;
							}
						}
					}

					double t1_min = Double.POSITIVE_INFINITY;
					double t1_max = -Double.POSITIVE_INFINITY;
					double t1_edge_dist = Double.POSITIVE_INFINITY;
					for(int i = 0; i < line1.size(); ++i){
						Point p = line1.get(i);
						double rt = dx1 * p.y + -dy1 * p.x;

						if(rt > t1_max){
							t1_max = rt;

							double ed = edgeDist(p, mat.width(), mat.height());
							if(ed < Math.abs(t1_edge_dist)){
								t1_edge_dist = ed;
							}
						}

						if(rt < t1_min){
							t1_min = rt;
							double ed = edgeDist(p, mat.width(), mat.height());
							if(ed < Math.abs(t1_edge_dist)){
								t1_edge_dist = -ed;
							}
						}
					}

					Logger.trace("LT " + t0_min + " " + t0_max + " " + t0_edge_dist);
					Logger.trace("LT " + t1_min + " " + t1_max + " " + t1_edge_dist);

					/* At this point, we know in which direction the bar leaves the camera view
					   We also have an underestimate of the end of the bar.

						 We now search for points within the bounds of the bar lines, but also
						 close to the end points. This is a fairly narrow band so whatever points
						 are found can be fed directly to the regression function.
					 */
        	List<Point> endline = new ArrayList<>();

					for(int i = 0; i < mat.rows(); ++i){
						for(int k = 0; k < mat.cols(); ++k){
							double v = mat.get(i, k)[0];
							if(v > 0){

								double rp0 = dx0 * k + dy0 * i;
								double rp1 = dx1 * k + dy1 * i;

								/* Check if point is in between lines */
								if(rp0 > (rho_0_avg + p_sf) && rp1 < (rho_1_avg - p_sf)){

									/* Check if point is close to the end of the bar. */
									double rt0 = dx0 * i + - dy0 * k;
									if(t0_edge_dist > 0){
										/* t1_min represents the end of the bar.*/
										if(rt0 > t0_min - p_sf && rt0 < t0_min + p_sf){
											Point p = new Point(k, i);
											endline.add(p);
										}
									}
									else{
										/* t1_min represents the end of the bar.*/
										if(rt0 > t0_max - p_sf && rt0 < t0_max + p_sf){
											Point p = new Point(k, i);
											endline.add(p);
										}
									}
								}
							}
						}
					}

					/* OK now let's regress this line. */
					MatOfPoint mline2 = new MatOfPoint();
					mline2.fromList(endline);
					Mat fline2 = new Mat();
					Imgproc.fitLine(mline2, fline2, Imgproc.CV_DIST_L2, 0, 0.01, 0.01);

					Logger.trace("EV " + fline2.get(2,0)[0] + " " + fline2.get(3,0)[0]);
					Logger.trace("EV A" + (Math.acos(fline2.get(0,0)[0]) / Math.PI * 180.0));

					Point eb = new Point(fline2.get(2,0)[0], fline2.get(3,0)[0]);
					Point ev = new Point(fline2.get(0,0)[0], fline2.get(1,0)[0]);

					Point l0b = new Point(fline0.get(2,0)[0], fline0.get(3,0)[0]);
					Point l0v = new Point(fline0.get(0,0)[0], fline0.get(1,0)[0]);

					Point l1b = new Point(fline1.get(2,0)[0], fline1.get(3,0)[0]);
					Point l1v = new Point(fline1.get(0,0)[0], fline1.get(1,0)[0]);

					/* Intercept the line with the other lines to find the bar endpoints */
					Point ep0 = calcLineLineIntersection(eb, ev, l0b, l0v);
					Point ep1 = calcLineLineIntersection(eb, ev, l1b, l1v);

					retval.add(renderLine(rho_0_avg, theta_0_avg, t0_min, t0_max));
					retval.add(renderLine(rho_1_avg, theta_1_avg, t1_min, t1_max));

        	List<Point> endlineFinal = new ArrayList<>();
					endlineFinal.add(ep0);
					endlineFinal.add(ep1);

					MatOfPoint eline = new MatOfPoint();
					eline.fromList(endlineFinal);
					retval.add(eline);

					/*
					MatOfPoint eline2 = new MatOfPoint();
					eline2.fromList(endline);
					retval.add(eline2);
					*/


					fline2.release();
				}

				fline0.release();
				fline1.release();
				output.release();
        return new Result(null, retval);
    }
}
