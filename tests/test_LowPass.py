import unittest
import numpy as np
from dadi.LowPass import LowPass
from math import factorial

class TestLowPassFunctions(unittest.TestCase):

    def test_compute_cov_dist(self):
        data_dict = {
            'entry1': {'coverage': {'pop1': [1, 2, 2, 3, 6], 'pop2': [2, 1, 3, 3, 1]}},
            'entry2': {'coverage': {'pop1': [1, 1, 2, 2, 3], 'pop2': [1, 3, 3, 6, 1]}},
        }
        pop_ids = ['pop1', 'pop2']
        expected_result = {
            'pop1': np.array([[0, 1, 2, 3, 4, 5, 6], [0, 0.3, 0.4, 0.2, 0, 0, 0.1]]),
            'pop2': np.array([[0, 1, 2, 3, 4, 5, 6], [0, 0.4, 0.1, 0.4, 0, 0, 0.1]])
        }
        result = LowPass.compute_cov_dist(data_dict, pop_ids)
        for pop in pop_ids:
            np.testing.assert_array_almost_equal(result[pop], expected_result[pop])

    def test_part_inbreeding_probability(self):
        parts = [[0, 0, 2], [1, 1, 0]]
        Fx = 0.1
        result = LowPass.part_inbreeding_probability(parts, Fx)

        expected_part_prob = np.array([])
        for part in parts:
            p = (2 * part.count(0) + part.count(1)) / (2 * len(part))
            q = 1 - p

            # Hardy-Weinberg with inbreeding
            p00 = p**2 + p*q*Fx
            p01 = 2*p*q*(1 - Fx)
            p11 = q**2 + p*q*Fx
            
            n, n00, n01, n11 = len(part), part.count(0), part.count(1), part.count(2)
            
            expected_part_prob = np.append(expected_part_prob, (factorial(n) / (factorial(n00) * factorial(n01) * factorial(n11))) * (p00 ** n00) * (p01 ** n01) * (p11 ** n11))

        expected_result = expected_part_prob / sum(expected_part_prob)

        np.testing.assert_array_almost_equal(result, expected_result)

    def test_partitions_and_probabilities(self):
        n_sequenced = 6
        partition_type = 'allele_frequency'
        Fx = 0
        allele_frequency = 2
        partitions, partition_probabilities = LowPass.partitions_and_probabilities(n_sequenced, partition_type, Fx, allele_frequency)

        expected_partitions = [[0, 0, 2], [0, 1, 1]]
        expected_partition_probabilities = np.array([0.2, 0.8])

        np.testing.assert_array_almost_equal(partitions, expected_partitions)
        np.testing.assert_array_almost_equal(partition_probabilities, expected_partition_probabilities)

    def test_split_list_by_lengths(self):
        input_list = [1, 2, 3, 4, 5, 6]
        lengths_list = [2, 2, 2]
        expected_result = [[1, 2], [3, 4], [5, 6]]
        result = LowPass.split_list_by_lengths(input_list, lengths_list)
        self.assertEqual(result, expected_result)

    def test_flatten_nested_list(self):
        nested_list = [[1, 5], [3, 4]]
        expected_result_concatenating = [4, 5, 8, 9]
        expected_result_multiplying = [3, 4, 15, 20]
        
        result_concatenating = LowPass.flatten_nested_list(nested_list, '+')
        result_multiplying = LowPass.flatten_nested_list(nested_list, '*')

        self.assertEqual(expected_result_concatenating, result_concatenating)
        self.assertEqual(expected_result_multiplying, result_multiplying)

    def test_simulate_reads(self):
        prob = [0 if x < 100 else 1 for x in range(1,101)]
        coverage_distribution = {
            'pop1': np.array([list(range(1,101)), prob])
        }
        flattened_partition = [0, 1, 2, 1, 0]
        
        pop_n_sequenced = [5]
        number_simulations = 10
        
        np.random.seed(42)
        n_ref, n_alt = LowPass.simulate_reads(coverage_distribution, flattened_partition, pop_n_sequenced, number_simulations)
        
        assert np.all(n_ref[:, 0] > 0)
        assert np.all(n_ref[:, 1] > 0)
        assert np.all(n_ref[:, 2] == 0)
        assert np.all(n_ref[:, 3] > 0)
        assert np.all(n_ref[:, 4] > 0)

        assert np.all(n_alt[:, 0] == 0)
        assert np.all(n_alt[:, 1] > 0)
        assert np.all(n_alt[:, 2] > 0)
        assert np.all(n_alt[:, 3] > 0)
        assert np.all(n_alt[:, 4] == 0)

    def test_subsample_genotypes_1D(self):
        genotype_calls = np.array([[99, 1, 2, 99], [1, 99, 0, 99], [99, 99, 99, 99]])
        n_subsampling = 4
        
        result = LowPass.subsample_genotypes_1D(genotype_calls, n_subsampling)
        result_sorted = np.array([sorted(subarray) for subarray in result])

        expected_result = np.array([[1, 2], [0, 1]])

        np.array_equal(result_sorted, expected_result)

    def test_simulate_GATK_multisample_calling(self):
        prob = [0 if x < 100 else 1 for x in range(1,101)]
        coverage_distribution = {
            'pop1': np.array([list(range(1,101)), prob])
        }
        
        allele_frequency = [1]
        n_sequenced = [4]
        n_subsampling = [4]
        number_simulations = 100
        Fx = [0]
        
        np.random.seed(42)
        result = LowPass.simulate_GATK_multisample_calling(coverage_distribution, allele_frequency, n_sequenced, n_subsampling, number_simulations, Fx)
        
        expected_result = np.array([0, 1, 0, 0, 0])

        np.array_equal(result, expected_result)

    def test_probability_of_no_call_1D_GATK_multisample(self):
        prob = [.0001 if x < 100 else .9999 for x in range(0, 101)]
        prob = [x/sum(prob) for x in prob]

        coverage_distribution = {
            'pop1': np.array([list(range(0, 101)), prob])
        }

        n_sequenced = 10
        Fx = 0
        
        result = LowPass.probability_of_no_call_1D_GATK_multisample(coverage_distribution['pop1'], n_sequenced, Fx)

        assert np.all(result[1:] < 1e-3)

    def test_probability_enough_individuals_covered(self):
        prob = [0 if x < 100 else 1 for x in range(1,101)]
        coverage_distribution = {
            'pop1': np.array([list(range(1,101)), prob])
        }

        n_sequenced = 4
        n_subsampling = 2
        result = LowPass.probability_enough_individuals_covered(coverage_distribution['pop1'], n_sequenced, n_subsampling)
        
        self.assertEqual(result, 1.0)

    def test_projection_inbreeding(self):
        partition = [0, 1, 2, 1]
        k = 2
        result = LowPass.projection_inbreeding(partition, k)
        self.assertEqual(result.shape[0], k + 1)

    def test_projection_matrix(self):
        n_sequenced = 4
        n_subsampling = 2
        Fx = 0
        result = LowPass.projection_matrix(n_sequenced, n_subsampling, Fx)
        self.assertEqual(result.shape, (n_sequenced + 1, n_subsampling + 1))

    def test_calling_error_matrix(self):
        prob = [0 if x < 100 else 1 for x in range(1,101)]
        coverage_distribution = np.array([list(range(1,101)), prob])

        n_subsampling = 2
        Fx = 0
        result = LowPass.calling_error_matrix(coverage_distribution, n_subsampling, Fx)
        diagonal_result = np.diag(result)

        assert np.allclose(diagonal_result, 1.0, atol=1e-9)

if __name__ == '__main__':
    unittest.main()
