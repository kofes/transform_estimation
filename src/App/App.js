import React, {useEffect} from 'react';
import jsfeat from 'jsfeat';
import img1 from '../images/photo_01.jpg';
import img2 from '../images/photo_02.jpg';
import img3 from '../images/photo_03.jpg';
import './App.css';

// central difference using image moments to find dominant orientation
const u_max = new Int32Array([15,15,15,15,14,14,14,13,13,12,11,10,9,8,6,3,0]);
function ic_angle(img, px, py) {
    var half_k = 15; // half patch size
    var m_01 = 0, m_10 = 0;
    var src=img.data, step=img.cols;
    var u=0, v=0, center_off=(py*step + px)|0;
    var v_sum=0,d=0,val_plus=0,val_minus=0;

    // Treat the center line differently, v=0
    for (u = -half_k; u <= half_k; ++u)
        m_10 += u * src[center_off+u];

    // Go line by line in the circular patch
    for (v = 1; v <= half_k; ++v) {
        // Proceed over the two lines
        v_sum = 0;
        d = u_max[v];
        for (u = -d; u <= d; ++u) {
            val_plus = src[center_off+u+v*step];
            val_minus = src[center_off+u-v*step];
            v_sum += (val_plus - val_minus);
            m_10 += u * (val_plus + val_minus);
        }
        m_01 += v * v_sum;
    }

    return Math.atan2(m_01, m_10);
}

// non zero bits count
function popcnt32(n) {
    n -= ((n >> 1) & 0x55555555);
    n = (n & 0x33333333) + ((n >> 2) & 0x33333333);
    return (((n + (n >> 4))& 0xF0F0F0F)* 0x1010101) >> 24;
}

function pairwise(arr, callback, skips) {
    let result = [];
    skips = skips || 1;
    for (let i = 0; i < arr.length - skips; ++i) {
        result.push(callback(arr[i], arr[i + skips], i, i + skips));
    }
    return result;
}

function render_corners(corners, count, img, step, color) {
    if (!color) {
        color = {
            r: 255,
            g: 0,
            b: 0
        }
    }
    var pix = ((0xff << 24) | (color.b << 16) | (color.g << 8) | color.r);
    for(var i=0; i < count; ++i)
    {
        var x = corners[i].x;
        var y = corners[i].y;
        var off = (x + y * step);
        img[off] = pix;
        img[off-1] = pix;
        img[off+1] = pix;
        img[off-step] = pix;
        img[off+step] = pix;
    }
}

function render_mono_image(src, dst, sw, sh, dw) {
    let alpha = (0xff << 24);
    for (let i = 0; i < sh; ++i) {
        for (let j = 0; j < sw; ++j) {
            let pix = src[i*sw+j];
            dst[i*dw+j] = alpha | (pix << 16) | (pix << 8) | pix;
        }
    }
}

function render_matches(ctx, arrayOfKeypoints, matches, color) {
    if (!matches) {
        return;
    }

    for (let i = 0; i < matches.length; ++i) {
        let m = matches[i];
        let kp1 = arrayOfKeypoints[0].keypoints[m.keypoint_index_1];
        let kp2 = arrayOfKeypoints[1].keypoints[m.keypoint_index_2];

        // only inliers
        ctx.strokeStyle = color ? color : "rgb(0,255,0)";

        ctx.beginPath();
        ctx.moveTo(kp1.x, kp1.y);
        ctx.lineTo(kp2.x, kp2.y);
        ctx.lineWidth=1;
        ctx.stroke();
    }
}

function App() {
    const timeout = 50;

    const images = [img1, img2, img3];
    const imagesObjects = [];

    const blur_size = 5;
    const lap_thres = 22;
    const eigen_thres = 14;
    const max_keypoints = 1000;
    const norm_threshold = 200;

    const canvas_width = 960;
    const canvas_height = 1280;

    const getCanvasByImageIndex = (index, type) => document.getElementById('image-' + (type ? (type + '-') : '') + (index !== undefined ? index : ''));
    const getContextByImageIndex = (index, type) => getCanvasByImageIndex(index, type).getContext('2d');

    let showImages = () => (<div className="images">
        {images.map((value, index) => (<canvas
            key={index}
            id={'image-' + index}
            width={canvas_width}
            height={canvas_height}
        />))}
    </div>);

    let showGrayScaleImages = () => (<div className="images gray">
        {images.map((value, index) => {
            return (<canvas
                key={index}
                id={'image-gray-' + index}
                width={canvas_width}
                height={canvas_height}
            ></canvas>)
        })}
    </div>);

    let showKeyPoints = () => (<div className="keypoints">
        {images.map((value, index) => {
            return (<canvas
                key={index}
                id={'image-with-keypoints-' + index}
                width={canvas_width}
                height={canvas_height}
            ></canvas>)
        })}
    </div>);

    let showCorrespondencePoints = () => (<div className="correspondeces">
        {images.map((val, index) => {
            if (!index) {
                return (<span key={index}/>);
            }
            return (<canvas
                    key={index}
                    id={'image-with-correspondence-points-' + (index-1)}
                    width={canvas_width}
                    height={canvas_height}
                />);
            })
        }
    </div>);

    // TODO: resolve match data {keypoints, count}

    // naive brute-force matching.
    // each on screen point is compared to all pattern points
    // to find the closest match
    function match(data1, data2) {
        let matches = [];

        let query_u32 = data1.descriptors.buffer.i32; // cast to integer buffer
        let qd_off = 0;

        for(let qidx = 0; qidx < data1.count; ++qidx) {
            let best_dist = 256;
            let best_dist2 = 256;
            let best_idx = -1;

            let best_norm = canvas_height * canvas_width;

            let ld_i32 = data2.descriptors.buffer.i32; // cast to integer buffer
            let ld_off = 0;

            for(let pidx = 0; pidx < data2.count; ++pidx) {
                let curr_d = 0;

                // our descriptor is 32 bytes so we have 8 Integers
                for (let k=0; k < 8; ++k) {
                    curr_d += popcnt32( query_u32[qd_off+k]^ld_i32[ld_off+k] );
                }

                if(curr_d < best_dist) {
                    best_dist2 = best_dist;
                    best_dist = curr_d;
                    best_idx = pidx;

                    // best_norm2 = best_norm;
                    best_norm = Math.sqrt(
                        Math.pow(data1.keypoints[qidx].x - data2.keypoints[best_idx].x, 2) +
                        Math.pow(data1.keypoints[qidx].y - data2.keypoints[best_idx].y, 2)
                    );
                } else if(curr_d < best_dist2) {
                    best_dist2 = curr_d;
                }

                ld_off += 8; // next descriptor
            }

            // filter out by some threshold
            // if(best_dist < options.match_threshold) {
            //     matches[num_matches].screen_idx = qidx;
            //     matches[num_matches].pattern_lev = best_lev;
            //     matches[num_matches].pattern_idx = best_idx;
            //     num_matches++;
            // }


            // filter using the ratio between 2 closest matches
            if(best_dist < best_dist2 && best_norm < norm_threshold) {
                matches.push({
                    keypoint_index_1: qidx,
                    keypoint_index_2: best_idx,
                });
            }

            qd_off += 8; // next query descriptor
        }
        // console.log(matches.length);

        return matches;
    }

    // estimate homography transform between matched points
    function find_transform(keypoints, matches) {
        if (matches && matches.length === 0) {
            return {};
        }
        // transform matrix
        let homo3x3 = new jsfeat.matrix_t(3, 3, jsfeat.F32C1_t);
        let match_mask = new jsfeat.matrix_t(matches.length, 1, jsfeat.U8_t | jsfeat.C1_t);

        // motion kernel
        let mm_kernel = new jsfeat.motion_model.homography2d();
        // ransac params
        let num_model_points = 4; // minimum points to estimate motion
        let reproj_threshold = 3; // max error to classify as inliner
        let eps = 0.5; // max outliers ratio
        let prob = 0.99; // probability of success
        let ransac_param = new jsfeat.ransac_params_t(num_model_points, reproj_threshold, eps, prob);

        let points1_xy = []; // screen (1)
        let points2_xy = []; // pattern (2)

        // construct correspondences
        for (let i = 0; i < matches.length; ++i) {
            let m = matches[i];
            let kp1 = keypoints[0].keypoints[m.keypoint_index_1];
            let kp2 = keypoints[1].keypoints[m.keypoint_index_2];
            points1_xy[i] = {"x": kp1.x, "y": kp1.y};
            points2_xy[i] = {"x": kp2.x, "y": kp2.y};
        }

        // estimate motion
        let max_iterations = 1000;
        let ok = jsfeat.motion_estimator.ransac(ransac_param, mm_kernel, points2_xy, points1_xy, matches.length, homo3x3, match_mask, max_iterations);

        let count_filtered_matches = 0;
        let inliers = [];
        let outliers = [];
        if (ok) {
            for (let i = 0; i < matches.length; ++i) {
                if (match_mask.data[i]) {
                    points1_xy[count_filtered_matches].x = points1_xy[i].x;
                    points1_xy[count_filtered_matches].y = points1_xy[i].y;

                    points2_xy[count_filtered_matches].x = points2_xy[i].x;
                    points2_xy[count_filtered_matches].y = points2_xy[i].y;

                    ++count_filtered_matches;

                    inliers.push(matches[i]);
                } else {
                    outliers.push(matches[i]);
                }
            }
            // run kernel directly with inliers only
            mm_kernel.run(points2_xy, points1_xy, homo3x3, count_filtered_matches);
        } else {
            jsfeat.matmath.identity_3x3(homo3x3, 1.0);
        }
        return {
            count: count_filtered_matches,
            homography: mm_kernel,
            transform: homo3x3,
            inliers,
            outliers
        };
    }

    function find_essential_matrix(keypoints, matches) {
        if (!matches) {
            return;
        }

        let essential_matrix = new jsfeat.matrix_t(3, 3, jsfeat.F32C1_t);

        const sample = matches
            .map(x => ({ x, r: Math.random() }))
            .sort((a, b) => a.r - b.r)
            .map(a => a.x)
            .slice(0, 9);

        const pairs_coordinates = [];
        for (let i = 0; i < sample.length; ++i) {
            let m = sample[i];
            let point1 = {
                x: keypoints[0].keypoints[m.keypoint_index_1].x,
                y: keypoints[0].keypoints[m.keypoint_index_1].y,
            };
            let point2 = {
                x: keypoints[1].keypoints[m.keypoint_index_2].x,
                y: keypoints[1].keypoints[m.keypoint_index_2].y,
            };
            pairs_coordinates.push([point1, point2]);
        }

        let e_matrix = new jsfeat.matrix_t(9, 1, jsfeat.F32C1_t);
        let b_matrix = new jsfeat.matrix_t(9, 1, jsfeat.F32C1_t);
        let m_matrix = new jsfeat.matrix_t(9, 8, jsfeat.F32C1_t);
        for (let i = 0; i < m_matrix.rows; ++i) {
            m_matrix.data[i * m_matrix.cols + 0] = pairs_coordinates[i][0].x * pairs_coordinates[i][1].x;
            m_matrix.data[i * m_matrix.cols + 1] = pairs_coordinates[i][0].y * pairs_coordinates[i][1].x;
            m_matrix.data[i * m_matrix.cols + 2] = pairs_coordinates[i][1].x;

            m_matrix.data[i * m_matrix.cols + 3] = pairs_coordinates[i][0].x * pairs_coordinates[i][1].y;
            m_matrix.data[i * m_matrix.cols + 4] = pairs_coordinates[i][0].y * pairs_coordinates[i][1].y;
            m_matrix.data[i * m_matrix.cols + 5] = pairs_coordinates[i][1].y;

            m_matrix.data[i * m_matrix.cols + 6] = pairs_coordinates[i][0].x;
            m_matrix.data[i * m_matrix.cols + 7] = pairs_coordinates[i][0].y;
            m_matrix.data[i * m_matrix.cols + 8] = 1;
        }

        return jsfeat.linalg.svd_solve(m_matrix, e_matrix, b_matrix);
    }

    function find_points_positions(matched_points, transforms) {
        if (!transforms || transforms.length < 2) {
            return;
        }
        for (let i = 0; i < transforms.length; ++i) {
            if (!transforms[i]) {
                return;
            }
        }
        let [transform1, transform2, transform3, ...anotherTransforms] = transforms;

        let get = (matrix, i, j) => {
            return matrix.data[i * matrix.cols + j];
        };
        let set = (matrix, i, j, val) => {
            matrix.data[i * matrix.cols + j] = val;
        };

        for (let i = 0; i < 1/*matched_points.length*/; ++i) {
            let t_matrix = new jsfeat.matrix_t(6, 4, jsfeat.F32C1_t);

            let [point1, point2, point3, ...anotherTransforms] = matched_points[i];
            // let [transform1, transform2] = transforms;

            // row 1
            for (let k = 0; k < 4; ++k) {
                set(t_matrix, 0, k,
                    point1.x * get(transform1, 2, k) - get(transform1, 0, k)
                );
            }
            // row 2
            for (let k = 0; k < 4; ++k) {
                set(t_matrix, 1, k,
                    point1.y * get(transform1, 2, k) - get(transform1, 1, k)
                );
            }

            // row 3
            for (let k = 0; k < 4; ++k) {
                set(t_matrix, 2, k,
                    point2.x * get(transform2, 2, k) - get(transform2, 0, k)
                );
            }
            // row 4
            for (let k = 0; k < 4; ++k) {
                set(t_matrix, 3, k,
                    point2.y * get(transform2, 2, k) - get(transform2, 1, k)
                );
            }


            // row 5
            for (let k = 0; k < 4; ++k) {
                set(t_matrix, 2, k,
                    point3.x * get(transform3, 2, k) - get(transform3, 0, k)
                );
            }
            // row 6
            for (let k = 0; k < 4; ++k) {
                set(t_matrix, 3, k,
                    point3.y * get(transform3, 2, k) - get(transform3, 1, k)
                );
            }


            console.log('---point 1---');
            console.log(point1);
            console.log('---point 2---');
            console.log(point2);
            console.log('---transform 1---');
            console.log(transform1);
            console.log('---transform 2---');
            console.log(transform2);
            console.log('---T---');
            console.log(t_matrix);

            let U = new jsfeat.matrix_t(t_matrix.rows, t_matrix.rows, jsfeat.F32C1_t);
            let W = new jsfeat.matrix_t(1, t_matrix.cols, jsfeat.F32C1_t);
            let V_t = new jsfeat.matrix_t(t_matrix.cols, t_matrix.cols, jsfeat.F32C1_t);

            jsfeat.linalg.svd_decompose(t_matrix, W, U, V_t, jsfeat.SVD_V_T);

            console.log('---W---');
            console.log(W);
            console.log('---U---');
            console.log(U);
            console.log('---V_t---');
            console.log(V_t);

            // let x_vector = new jsfeat.matrix_t(4, 1, jsfeat.F32C1_t);
            // jsfeat.linalg.svd_decompose()
            // jsfeat.linalg.svd_solve(t_matrix, x_vector, new jsfeat.matrix_t(4, 1, jsfeat.F32C1_t));
            // console.log('---solution---');
            // console.log(x_vector);
        }
        // jsfeat.linalg.svd_solve(t_matrix, x_vector, new jsfeat.matrix_t(4, 1, jsfeat.F32C1_t));
        // console.log(x_vector);
    }
/*
    0: 745.9229125976562
    1: -13.39715576171875
    2: 737.67724609375
    3: -730.1497192382812
    4: 619.478759765625
    5: -13.746614456176758
    6: 629.9678955078125
    7: -640.0626220703125
    8: 742.1522216796875
    9: -17.949859619140625
    10: 754.7247314453125
    11: -766.8300170898438
    12: 537.6793823242188
    13: -11.862923622131348
    14: 546.7831420898438
    15: -555.5443725585938
*/
    function tick() {
        const imagesData = images.map((_, index) => {
            const canvas = getCanvasByImageIndex(index);
            return getContextByImageIndex(index).getImageData(0, 0, canvas.width, canvas.height);
        });

        const grayScaledImages = imagesData.map((data, index) => {
            const canvas = getCanvasByImageIndex(index);
            let img_u8 = new jsfeat.matrix_t(canvas.width, canvas.height, jsfeat.U8_t | jsfeat.C1_t);
            jsfeat.imgproc.grayscale(data.data, canvas.width, canvas.height, img_u8);
            return img_u8;
        });

        grayScaledImages.map((img_u8, index) => {
            const canvas = getCanvasByImageIndex(index, 'gray');
            const context = getContextByImageIndex(index, 'gray');
            const data = context.getImageData(0, 0, canvas.width, canvas.height);
            let data_u32 = new Uint32Array(data.data.buffer);
            render_mono_image(img_u8.data, data_u32, img_u8.cols, img_u8.rows, canvas.width);
            context.putImageData(data, 0, 0);
            return true;
        });

        let keyPoints = imagesData.map((data, index) => {
            let img_u8 = new jsfeat.matrix_t(data.width, data.height, jsfeat.U8_t | jsfeat.C1_t);
            let imgSmooth_u8 = new jsfeat.matrix_t(data.width, data.height, jsfeat.U8_t | jsfeat.C1_t);

            // from rgb to grayscale
            jsfeat.imgproc.grayscale(data.data, data.width, data.height, img_u8);

            // from grayscale to `blured` grayscale
            jsfeat.imgproc.gaussian_blur(img_u8, imgSmooth_u8, blur_size|0);

            // detect keypoints from `blured` grayscale
            jsfeat.yape06.laplacian_threshold = lap_thres|0;
            jsfeat.yape06.min_eigen_value_threshold = eigen_thres|0;

            const algo = jsfeat.yape06;
            // algo.init(canvas_width, canvas_height);

            let keypoints = [];
            for (let i = 0; i < data.width * data.height; ++i) {
                keypoints[i] = new jsfeat.keypoint_t(0,0,0,0,-1);
            }
            let countKeypoints = algo.detect(imgSmooth_u8, keypoints, 35);

            // sort keypoints by score
            if (countKeypoints > max_keypoints) {
                jsfeat.math.qsort(keypoints, 0, countKeypoints-1, (a, b) => (b.score < a.score));
                countKeypoints = max_keypoints;
            }

            // calculate dominant orientation for each keypoint
            for (let i = 0; i < countKeypoints; ++i) {
                keypoints[i].angle = ic_angle(imgSmooth_u8, keypoints[i].x, keypoints[i].y);
            }

            let descriptors = new jsfeat.matrix_t(32, countKeypoints, jsfeat.U8_t | jsfeat.C1_t);
            // describe descriptors
            jsfeat.orb.describe(imgSmooth_u8, keypoints, countKeypoints, descriptors);

            return {
                keypoints,
                descriptors,
                count: countKeypoints
            };
        });

        // draw keypoints
        imagesData.map((data, index) => {
            const canvas = getCanvasByImageIndex(index, 'with-keypoints');
            const context = getContextByImageIndex(index, 'with-keypoints');

            const drawData = context.getImageData(0, 0, canvas.width, canvas.height);
            let data_u32 = new Uint32Array(drawData.data.buffer);

            let color = {
                r: (index === 0 ? 255 : 0),
                g: (index === 1 ? 255 : 0),
                b: (index === 2 ? 255 : 0)
            };
            render_corners(keyPoints[index].keypoints, keyPoints[index].count, data_u32, canvas.width, color);
            context.putImageData(drawData, 0, 0);
        });

        // draw backgrounds mix
        pairwise(keyPoints, (data1, data2, idx1, idx2) => {
            // if (idx1 > 0) {
            //     return;
            // }
            let canvas = getCanvasByImageIndex(idx1, 'with-correspondence-points');
            let ctx = getContextByImageIndex(idx1, 'with-correspondence-points');

            ctx.clearRect(0, 0, canvas.width, canvas.height);

            let images = imagesObjects;
            ctx.globalAlpha = 0.5;
            [idx1, idx2].forEach(idx => {
                let val = images[idx];
                ctx.drawImage(val, 0, 0, canvas.width, canvas.height);
            });
            ctx.globalAlpha = 1.0;
        });

        // calculate matches
        let matches = pairwise(keyPoints, (data1, data2, idx1, idx2) => {
            let matches = match(data1, data2);
            return {
                keypoints: [data1, data2],
                matches
            };
        });

        matches.forEach(({keypoints, matches}, index) => {
            // if (index > 0) {
            //     return;
            // }
            let canvas = getCanvasByImageIndex(index, 'with-correspondence-points');
            let ctx = getContextByImageIndex(index, 'with-correspondence-points');

            const drawData = ctx.getImageData(0, 0, canvas.width, canvas.height);
            let data_u32 = new Uint32Array(drawData.data.buffer);

            render_corners(keypoints[0].keypoints, keypoints[0].count, data_u32, canvas.width, {r: 255, g: 0, b: 0});
            render_corners(keypoints[1].keypoints, keypoints[1].count, data_u32, canvas.width, {r: 0, g: 255, b: 0});
            ctx.putImageData(drawData, 0, 0);
            // render_matches(ctx, keypoints, matches, {r: 255, g: 255, b: 255});
        });

        if (!matches) {
            return;
        }
        // draw matches
        let transformData = matches.map(({keypoints, matches}) => {
            return {
                data: [],
                keypoints,
                matches
            };
        });

        for (let iteration = 0; iteration < 2; ++iteration) {
            // calculate new inliers
            transformData = transformData.map(({data, keypoints, matches}) => {
                let currentMatches = data.length > 0 ? data[data.length - 1].outliers : matches;
                if (!currentMatches) {
                    return {
                        data,
                        keypoints,
                        matches
                    };
                }

                let bestTransformData = find_transform(keypoints, currentMatches);
                for (let i = 0; i < 20 * (iteration*2+1); ++i) {
                    let data = find_transform(keypoints, currentMatches);
                    if (data.count > bestTransformData.count) {
                        bestTransformData = data;
                    }
                }
                // console.log(bestTransformData);

                return {
                    'data': [...data, bestTransformData],
                    keypoints,
                    matches
                };
            });
        }

        // draw inliers
        transformData.map(({data, keypoints, matches}, index) => {
            // if (index > 0) {
            //     return;
            // }
            let ctx = getContextByImageIndex(index, 'with-correspondence-points');
            for (let i = 0; i < data.length; ++i) {
                let buffer = data[i];
                if (!buffer) {
                    continue;
                }
                let color = "rgb(0, 0, 255)";
                if (i === 1) {
                    color = "rgb(0, 255, 255)";
                }
                if (i === 2) {
                    color = "rgb(255, 0, 255)";
                }
                render_matches(ctx, keypoints, buffer.inliers, color);
            }
        });
        // show count of  same inliers
        let commonInliersData = [];
        let commonInliersData_t = function() {
            return {
                data_index_1: 0,
                data_index_2: 0,
                matches: []
            };
        }
        /*
         * [index] -> {
         * transformData[index]_data_index_1: number,
         * transformData[index+1]_data_index_2: number,
         * matches: [ transformData[index]_data[transformData[index]_data_index_1]_inliers_index, ...]
         * }
         *
        **/
        transformData.reduce((prevData, currData, index) => {
            let common_inliers = [];
            {
                let {data, keypoints, matches} = prevData;
                for (let k = 0; k < data.length; ++k) {
                    let inliersData = new commonInliersData_t();
                    inliersData.data_index_1 = k;

                    if (!data[k].inliers) {
                        common_inliers.push(inliersData);
                        continue;
                    }

                    // let res = [];
                    for (let i = 0; i < data[k].inliers.length; ++i) {
                        let m = data[k].inliers[i];
                        // let kp = keypoints[1].keypoints[m.keypoint_index_2];

                        inliersData.matches.push({
                            inliers_index_1: i
                        });

                        // res.push({
                        //     x: kp.x,
                        //     y: kp.y
                        // });
                    }


                    common_inliers.push(inliersData);
                }
                // return res;
            }

            let newCommonInliners = [];
            for (let l = 0; l < common_inliers.length; ++l) {
                let inliersData = common_inliers[l];

                let {data, keypoints, matches} = currData;
                for (let k = 0; k < data.length; ++k) {

                    let newInliersData = new commonInliersData_t();
                    newInliersData.data_index_1 = inliersData.data_index_1;
                    newInliersData.data_index_2 = k;

                    if (!data[k].inliers) {
                        newCommonInliners.push(newInliersData);
                        continue;
                    }


                    for (let i = 0; i < data[k].inliers.length; ++i) {
                        let m = data[k].inliers[i];

                        let kp = keypoints[0].keypoints[m.keypoint_index_1];

                        for (let j = 0; j < inliersData.matches.length; ++j) {
                            let inliersMatch = inliersData.matches[j];
                            let p = prevData.keypoints[1].keypoints[prevData.data[inliersData.data_index_1].inliers[inliersMatch.inliers_index_1].keypoint_index_2];
                            // let p = prevData.keypoints[1].keypoints[inliersMatch.keypoint_index_1];

                            let dist = Math.sqrt(Math.pow(p.x - kp.x, 2) + Math.pow(p.y - kp.y, 2));

                            if (dist < 1) {
                                newInliersData.matches.push({
                                    inliers_index_1: inliersMatch.inliers_index_1,
                                    inliers_index_2: i
                                });
                            }
                        }
                    }

                    newCommonInliners.push(newInliersData);
                }
            }

            // // filter not matched data
            // for (let l = 0; l < common_inliers.length; ++l) {
            //     let inliersData = common_inliers[l];
            //     let filtered_matches = [];
            //     for (let j = 0; j < inliersData.matches.length; ++j) {
            //         let inliersMatch = inliersData.matches[j];
            //         if (inliersMatch.hasOwnProperty('inliers_index_2')) {
            //             filtered_matches.push(inliersMatch);
            //         }
            //     }
            //     common_inliers[l].matches = filtered_matches;
            // }

            commonInliersData.push(newCommonInliners);

            // {
            //     let res = [];
            //     let {data, keypoints, matches} = currData;
            //     for (let k = 0; k < data.length; ++k) {
            //         for (let i = 0; i < data[k].inliers.length; ++i) {
            //             let m = data[k].inliers[i];
            //             let kp = keypoints[0].keypoints[m.keypoint_index_1];
            //             // let kp = keypoints[1].keypoints[m.keypoint_index_2];
            //             for (let j = 0; j < common_inliers.length; ++j) {
            //                 let p = common_inliers[j];
            //                 let dist = Math.sqrt(Math.pow(p.x - kp.x, 2) + Math.pow(p.y - kp.y, 2));
            //                 if (dist < 1) {
            //                     res.push({
            //                         x: p.x,
            //                         y: p.y
            //                     });
            //                 }
            //             }
            //         }
            //     }
            //     common_inliers = [...res];
            //     return res;
            // }
        });

        let maxCommonInliersData = commonInliersData.map((arr, index) => {
            let resultInd = 0;
            for (let i = 1; i < arr.length; ++i) {
                if (arr[i].matches.length > arr[resultInd].matches.length) {
                    resultInd = i;
                }
            }
            return arr[resultInd];
        });

        // console.log(maxCommonInliersData);

        //
        // let essential_matrix = new jsfeat.matrix_t(3, 3, jsfeat.F32C1_t);

        // // WARNING: very unstable - use homography(transform) matrix!
        // // 8-point algorithm
        // transformData.map(({data, keypoints, matches}) => {
        //     return find_essential_matrix(keypoints, data[0].inliers);
        // });

        let essential_matrices = transformData.map(({data, keypoints, matches}, index) => {
            let data_index = 0;
            if (index === maxCommonInliersData.length) {
                data_index = maxCommonInliersData[index-1].data_index_2;
            } else if (index < maxCommonInliersData.length) {
                data_index = maxCommonInliersData[index].data_index_1;
            }
            return data[data_index].transform;
        });

        let decompositions = essential_matrices.map(essential_matrix => {
            if (!essential_matrix) {
                return;
            }
            let U = new jsfeat.matrix_t(essential_matrix.rows, essential_matrix.rows, jsfeat.F32C1_t);
            let W = new jsfeat.matrix_t(1, essential_matrix.cols, jsfeat.F32C1_t);
            let V_t = new jsfeat.matrix_t(essential_matrix.cols, essential_matrix.cols, jsfeat.F32C1_t);
            jsfeat.linalg.svd_decompose(essential_matrix, W, U, V_t, jsfeat.SVD_V_T);
            return {
                U,
                W,
                V_t
            };
        });

        let transform4x3 = decompositions.map(val => {
            if (!val) {
                return;
            }
            let {U, W, V_t} = val;
            let H = new jsfeat.matrix_t(3, 3, jsfeat.F32C1_t);
            H.data[1] = -1;
            H.data[3] = 1;
            H.data[8] = 1;

            let R_1 = new jsfeat.matrix_t(3, 3, jsfeat.F32C1_t);
            jsfeat.matmath.multiply(R_1, U, H);
            jsfeat.matmath.multiply(R_1, R_1, V_t);

            let H_t = new jsfeat.matrix_t(3, 3, jsfeat.F32C1_t);
            jsfeat.matmath.transpose(H_t, H);

            let R_2 = new jsfeat.matrix_t(3, 3, jsfeat.F32C1_t);
            jsfeat.matmath.multiply(R_2, U, H_t);
            jsfeat.matmath.multiply(R_1, R_1, V_t);

            let U_t = new jsfeat.matrix_t(3, 3, jsfeat.F32C1_t);
            jsfeat.matmath.transpose(U_t, U);

            let c_1 = new jsfeat.matrix_t(3, 1, jsfeat.F32C1_t);
            let c_2 = new jsfeat.matrix_t(3, 1, jsfeat.F32C1_t);

            c_1.data[0] = U_t.data[6];
            c_1.data[1] = U_t.data[7];
            c_1.data[2] = U_t.data[8];

            c_2.data[0] = -c_1.data[0];
            c_2.data[1] = -c_1.data[1];
            c_2.data[2] = -c_1.data[2];

            // solutions: [R_1|-R_1*c_1],[R_1|-R_1*c_2],[R_2|-R_2*c_1],[R_2|-R_2*c_2]

            let R = R_1;
            let c = c_1;

            let transform4x3 = new jsfeat.matrix_t(4, 3, jsfeat.F32C1_t);
            transform4x3.data[0] = R.data[0];
            transform4x3.data[1] = R.data[1];
            transform4x3.data[2] = R.data[2];

            transform4x3.data[4] = R.data[3];
            transform4x3.data[5] = R.data[4];
            transform4x3.data[6] = R.data[5];

            transform4x3.data[8] = R.data[6];
            transform4x3.data[9] = R.data[7];
            transform4x3.data[10] = R.data[8];

            let t = new jsfeat.matrix_t(3, 1, jsfeat.F32C1_t);

            // console.log('----R----');
            // console.log(R);
            // console.log('----c----');
            // console.log(c);

            let c_t = new jsfeat.matrix_t(1, 3, jsfeat.F32C1_t);
            jsfeat.matmath.transpose(c_t, c);
            jsfeat.matmath.multiply(t, R, c_t);
            // console.log('----t----');
            // console.log(t);

            transform4x3.data[3] = -t.data[0];
            transform4x3.data[7] = -t.data[1];
            transform4x3.data[11] = -t.data[2];

            return transform4x3;
        });

        // console.log(transform4x3);
        for (let i = 0; i < transform4x3.length - 2; ++i) {
            if (!transform4x3[i] || !transform4x3[i+1] || !transform4x3[i+2]) {
                continue;
            }


            find_points_positions(matchedPoints, [transform4x3[i], transform4x3[i+1], transform4x3[i+2]]);
        }
        transform4x3.reduce((transform_matrix1, transform_matrix2, ind) => {
            if (!transform_matrix1 || !transform_matrix2) {
                return;
            }
            let common_inliers_data = maxCommonInliersData[ind-1];

            let prevData = transformData[ind-1];
            let currData = transformData[ind];

            let matchedPoints = [];
            for (let i = 0; i < common_inliers_data.matches.length; ++i) {
                let inliersMatch = common_inliers_data.matches[i];

                let p1 = prevData.keypoints[0].keypoints[prevData.data[common_inliers_data.data_index_1].inliers[inliersMatch.inliers_index_1].keypoint_index_1];

                let p2 = currData.keypoints[0].keypoints[currData.data[common_inliers_data.data_index_2].inliers[inliersMatch.inliers_index_2].keypoint_index_1];

                matchedPoints.push({
                    point1: {
                        x: p1.x,
                        y: p1.y
                    },
                    point2: {
                        x: p2.x,
                        y: p2.y
                    }
                });
            }

            find_points_positions(matchedPoints, {transform1: transform_matrix1, transform2: transform_matrix2});
        });
        // let inliers_positions = transform4x3.map((transform_matrix, ind) => {
        //     if (!transform_matrix) {
        //         return;
        //     }
        //     let { data, keypoints, matches } = transformData[ind];
        //     find_points_positions([keypoints[0].keypoints, keypoints[1].keypoints], data[0].inliers, transform_matrix);
        // });
    }

    useEffect(() => {
        // load images to canvas with/without keypoints
        const loadImage = (canvas, context, value) => {
            const img = new Image();
            img.src = value;
            img.onload = function() {
                // context.drawImage(img, 0, 0, this.width, this.height);
                // context.canvas.width = this.width;
                // context.canvas.height = this.height;
                context.drawImage(img, 0, 0, this.width, this.height);
            };
            imagesObjects.push(img);
        }

        [undefined, 'with-keypoints'].forEach(type => {
            images.forEach((value, index) => {
                const canvas = getCanvasByImageIndex(index, type);
                const context = getContextByImageIndex(index, type);
                loadImage(canvas, context, value);
            });
        });
        // const canvas = getCanvasByImageIndex(undefined, 'with-correspondence-points');
        // const context = getContextByImageIndex(undefined, 'with-correspondence-points');
        // loadImage(canvas, context, images[0]);

        // set tick interval
        const interval = setInterval(() => {tick()}, timeout);
        return () => clearInterval(interval);
    });

    return (
        <div className="App">
            {showImages()}
            {showGrayScaleImages()}
            {showKeyPoints()}
            {showCorrespondencePoints()}
        </div>
    );
}

export default App;
