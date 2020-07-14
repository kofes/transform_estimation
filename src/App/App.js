import React, {useEffect} from 'react';
import jsfeat from 'jsfeat';
import img1 from '../images/1_002.jpg';
import img2 from '../images/1_003.jpg';
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

function render_corners(corners, count, img, step, type) {
    var pix = type ? ((0xff << 24) | (0x00 << 16) | (0x00 << 8) | 0xff) : ((0xff << 24) | (0x00 << 16) | (0xff << 8) | 0x00);
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

    const images = [img1, img2];
    const imagesObjects = [];

    const blur_size = 7;
    const lap_thres = 22;
    const eigen_thres = 14;
    const max_keypoints = 1000;
    const norm_threshold = 60;

    const getCanvasByImageIndex = (index, type) => document.getElementById('image-' + (type ? (type + '-') : '') + (index !== undefined ? index : ''));
    const getContextByImageIndex = (index, type) => getCanvasByImageIndex(index, type).getContext('2d');

    let showImages = () => (<div className="images">
        {images.map((value, index) => (<canvas
            key={index}
            id={'image-' + index}
            width={640}
            height={480}
        />))}
    </div>);

    let showGrayScaleImages = () => (<div className="images gray">
        {images.map((value, index) => {
            return (<canvas
                key={index}
                id={'image-gray-' + index}
                width={640}
                height={480}
            ></canvas>)
        })}
    </div>);

    let showKeyPoints = () => (<div className="keypoints">
        {images.map((value, index) => {
            return (<canvas
                key={index}
                id={'image-with-keypoints-' + index}
                width={640}
                height={480}
            ></canvas>)
        })}
    </div>);

    let showCorrespondencePoints = () => {
        return (<canvas
            id={'image-with-correspondence-points-'}
            width={640}
            height={480}
        />)
    };

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

            let best_norm = 480.0*640;

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
            if(best_dist < 0.75*best_dist2 && best_norm < norm_threshold) {
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
        let num_model_points = 8; // minimum points to estimate motion
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

    function find_points_positions(keypoints, matches, transform) {
        if (!transform) {
            return;
        }
        const pairs_coordinates = [];
        for (let i = 0; i < matches.length; ++i) {
            let m = matches[i];
            let point1 = {
                x: keypoints[0][m.keypoint_index_1].x,
                y: keypoints[0][m.keypoint_index_1].y,
            };
            let point2 = {
                x: keypoints[1][m.keypoint_index_2].x,
                y: keypoints[1][m.keypoint_index_2].y,
            };
            pairs_coordinates.push([point1, point2]);
        }

        let get = (matrix, i, j) => {
            return matrix.data[i * matrix.cols + j];
        };
        let set = (matrix, i, j, val) => {
            matrix.data[i * matrix.cols + j] = val;
        };

        let t_matrix = new jsfeat.matrix_t(4, 4, jsfeat.F32C1_t);
        for (let i = 0; i < pairs_coordinates.length; ++i) {
            let [point1, point2] = pairs_coordinates[i];
            let [transform1, transform2] = [transform, transform];

            // row 1
            for (let k = 0; k < 4; ++k) {
                set(t_matrix, 0, k,
                    point1.x * get(transform1, 2, k) - get(transform1, 0, k)
                );
            }
            // row 2
            for (let k = 0; k < 4; ++k) {
                set(t_matrix, 1, k,
                    point1.y * get(transform2, 2, k) - get(transform2, 1, k)
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
        }
        let x_vector = new jsfeat.matrix_t(4, 1, jsfeat.F32C1_t);
        jsfeat.linalg.svd_solve(t_matrix, x_vector, new jsfeat.matrix_t(4, 1, jsfeat.F32C1_t));
        // console.log(x_vector);
    }

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

            let keypoints = [];
            for (let i = 0; i < data.width * data.height; ++i) {
                keypoints[i] = new jsfeat.keypoint_t(0,0,0,0,-1);
            }
            let countKeypoints = jsfeat.yape06.detect(imgSmooth_u8, keypoints, 35);

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

            render_corners(keyPoints[index].keypoints, keyPoints[index].count, data_u32, canvas.width, (index+1) % 2);
            context.putImageData(drawData, 0, 0);
        });

        // draw backgrounds mix
        pairwise(keyPoints, _ => {
            let canvas = getCanvasByImageIndex(undefined, 'with-correspondence-points');
            let ctx = getContextByImageIndex(undefined, 'with-correspondence-points');

            ctx.clearRect(0, 0, canvas.width, canvas.height);

            let images = imagesObjects;
            ctx.globalAlpha = 0.5;
            images.forEach((val) => {
                ctx.drawImage(val, 0, 0, canvas.width, canvas.height);
            });
            ctx.globalAlpha = 1.0;
        })

        // calculate matches
        let matches = pairwise(keyPoints, (data1, data2, idx1, idx2) => {
            let matches = match(data1, data2);
            return {
                keypoints: [data1, data2],
                matches
            };
        });

        matches.forEach(({keypoints, matches}) => {
            let canvas = getCanvasByImageIndex(undefined, 'with-correspondence-points');
            let ctx = getContextByImageIndex(undefined, 'with-correspondence-points');

            const drawData = ctx.getImageData(0, 0, canvas.width, canvas.height);
            let data_u32 = new Uint32Array(drawData.data.buffer);
            render_corners(keypoints[0].keypoints, keypoints[0].count, data_u32, canvas.width, true);
            render_corners(keypoints[1].keypoints, keypoints[1].count, data_u32, canvas.width, false);
            ctx.putImageData(drawData, 0, 0);
            render_matches(ctx, keypoints, matches);
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

        for (let iteration = 0; iteration < 3; ++iteration) {
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
        transformData.map(({data, keypoints, matches}) => {
            let ctx = getContextByImageIndex(undefined, 'with-correspondence-points');
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
        //
        // let essential_matrix = new jsfeat.matrix_t(3, 3, jsfeat.F32C1_t);

        // // WARNING: very unstable - use homography(transform) matrix!
        // // 8-point algorithm
        // transformData.map(({data, keypoints, matches}) => {
        //     return find_essential_matrix(keypoints, data[0].inliers);
        // });
        let essential_matrices = transformData.map(({data, keypoints, matches}) => {
            return data[0].transform;
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
            jsfeat.matmath.multiply(t, R, c);

            transform4x3.data[3] = -t.data[0];
            transform4x3.data[7] = -t.data[1];
            transform4x3.data[11] = -t.data[2];

            return transform4x3;
        });

        let inliers_positions = transform4x3.map((transform_matrix, ind) => {
            if (!transform_matrix) {
                return;
            }
            let { data, keypoints, matches } = transformData[ind];
            find_points_positions([keypoints[0].keypoints, keypoints[1].keypoints], data[0].inliers, transform_matrix);
        });
    }

    useEffect(() => {
        // load images to canvas with/without keypoints
        const loadImage = (canvas, context, value) => {
            const img = new Image();
            img.src = value;
            img.onload = () => {
                context.drawImage(img, 0, 0, canvas.width, canvas.height);
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
