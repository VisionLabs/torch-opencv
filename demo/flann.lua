local cv = require 'cv'
cv.flann = require 'cv.flann'
require 'cv.imgproc'
require 'cv.highgui'

-- generate data
local N_samples, N_features, fieldSize, N_query = 300, 2, 600, 20
local samples = torch.rand(N_samples, N_features):float() * fieldSize

-- create the index
local params = cv.flann.KDTreeIndexParams{trees=10}
local index = cv.flann.Index{samples, params, cv.FLANN_DIST_EUCLIDEAN}

-- generate query point
local query = torch.rand(1, N_features):float() * fieldSize
local indices, dists = index:knnSearch{query, N_query}

-- visualize data
local img = torch.ByteTensor(fieldSize, fieldSize, 3) * 0 + 180

-- draw query point
cv.circle{img, {query[1][1], query[1][2]}, 4, {0, 150, 0}, 7}
cv.circle{img, {query[1][1], query[1][2]}, 2, {0, 0  , 0}, 2}

-- draw data points
for i = 1,N_samples do
    cv.circle{img, {samples[i][1], samples[i][2]}, 2, {0, 0, 255}, 2}
end

-- draw search result
for i = 1, N_query do
    local datum = {samples[indices[1][i]][1], samples[indices[1][i]][2]}
    cv.circle{img, datum, 5, {200, 0, 10}, 2}
end

cv.imshow{"Red: data, green: query, blue: "..N_query.." approximate NNs", img}
cv.waitKey{0}