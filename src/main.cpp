//=======================================================================
// Copyright (c) 2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <iostream>
#include <chrono>
#include <atomic>

#include "dll/neural/dense_layer.hpp"
#include "dll/dbn.hpp"
#include "dll/trainer/stochastic_gradient_descent.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

namespace {

constexpr const size_t K = 3;

template<typename C>
float distance_knn(const C& lhs, const C& rhs){
    return etl::sum((lhs - rhs) >> (lhs - rhs));
}

struct distance_t {
    float dist;
    uint32_t label;
};

size_t vote_knn(const std::vector<distance_t>& distances){
    size_t votes[10]{};

    for(size_t k = 0; k < K; ++k){
        ++votes[distances[k].label];
    }

    size_t label = 0;

    for(size_t k = 1; k < 10; ++k){
        if(votes[k] > votes[label]){
            label = k;
        }
    }

    return label;
}

template<typename C, typename L>
double evaluate_knn(const C& training, const C& test, const L& training_labels, const L& test_labels){
    std::atomic<size_t> correct;

    correct = 0;

    cpp::default_thread_pool<> pool(8);

    cpp::parallel_foreach_n(pool, 0, test.size(), [&](const size_t i){
        std::vector<distance_t> distances(training.size());

        for(size_t j = 0; j < training.size(); ++j){
            float d = distance_knn(test[i], training[j]);
            distances[j] = {d, training_labels[j]};
        }

        std::sort(distances.begin(), distances.end(), [](const distance_t& lhs, const distance_t& rhs){
            return lhs.dist < rhs.dist;
        });

        if(vote_knn(distances) == test_labels[i]){
            ++correct;
        }
    });

    return correct / double(test.size());
}

} // end of anonymous

int main(int argc, char* argv []) {
    std::string model = "raw";
    if(argc > 1){
        model = argv[1];
    }

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_vector<float>>();

    if(model == "raw"){
        double accuracy = evaluate_knn(dataset.training_images, dataset.test_images, dataset.training_labels, dataset.test_labels);
        std::cout << "Raw: " << accuracy << std::endl;
    } else if(model == "test"){
        mnist::binarize_dataset(dataset);

        using dbn_t = dll::dbn_desc<
            dll::dbn_layers<
            dll::dense_desc<28 * 28, 500>::layer_t,
            dll::dense_desc<500, 250>::layer_t,
            dll::dense_desc<250, 10, dll::activation<dll::function::SOFTMAX>>::layer_t>,
            dll::momentum, dll::batch_size<100>, dll::trainer<dll::sgd_trainer>>::dbn_t;

        auto dbn = std::make_unique<dbn_t>();

        dbn->learning_rate = 0.1;
        dbn->initial_momentum = 0.9;
        dbn->momentum = 0.9;
        dbn->goal = -1.0; // Don't stop

        dbn->display();

        auto ft_error = dbn->fine_tune(dataset.training_images, dataset.training_labels, 50);
        std::cout << "ft_error:" << ft_error << std::endl;

        auto test_error = dll::test_set(dbn, dataset.test_images, dataset.test_labels, dll::predictor());
        std::cout << "test_error:" << test_error << std::endl;
    }

    return 0;
}
