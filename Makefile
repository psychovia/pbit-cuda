
#*
#* Compiler and graph-coloring settings
# 

NVCC ?= /usr/local/cuda-13.4/bin/nvcc
PYTHON ?= python
CONDA_PREFIX ?= /home/psychovia/miniforge3/envs/pbit-cugraph

CPU_FLAGS = -O3 -std=c++17 -I .
CUGRAPH_FLAGS = -O3 -std=c++20 -I . -DUSE_CUGRAPH -Wno-deprecated-declarations

LIB_FLAGS = -I "$(CONDA_PREFIX)/include" -L "$(CONDA_PREFIX)/lib" \
	-Xlinker -rpath -Xlinker "$(CONDA_PREFIX)/lib" \
	-lcugraph -lrmm -lcusolver -lcublas

ALGORITHMS = greedy ldf random smallest_last dsatur cugraph
GRAPHS = sparse dense

VERSIONS = $(foreach graph,$(GRAPHS),$(foreach algo,$(ALGORITHMS),executable/pbit_coloring_$(algo)_$(graph)))


#*
#* Graph-coloring build rules
#

.PHONY: all sparse dense simulate help

all: $(VERSIONS)

sparse: executable/sparse
dense: executable/dense

executable/graph_%.h: pbit-graphs/J_%.npy pbit-graphs/h_%.npy scripts/prepare_graph.py
	$(PYTHON) scripts/prepare_graph.py --J pbit-graphs/J_$*.npy \
		--h pbit-graphs/h_$*.npy --output $@

define CPU_RULE
executable/pbit_coloring_$(1)_%: cuda/pbit_coloring_benchmark.cu executable/graph_%.h Makefile
	$$(NVCC) $$(CPU_FLAGS) '-DGRAPH_HEADER="executable/graph_$$*.h"' \
		-DCOLORING_FUNCTION=$(2) $$< -o $$@
endef

$(eval $(call CPU_RULE,greedy,greedy_coloring))
$(eval $(call CPU_RULE,ldf,largest_degree_first_coloring))
$(eval $(call CPU_RULE,random,random_coloring))
$(eval $(call CPU_RULE,smallest_last,smallest_last_coloring))
$(eval $(call CPU_RULE,dsatur,dsatur_coloring))

executable/pbit_coloring_cugraph_%: cuda/pbit_coloring_benchmark.cu executable/graph_%.h Makefile
	$(NVCC) $(CUGRAPH_FLAGS) '-DGRAPH_HEADER="executable/graph_$*.h"' \
		$< $(LIB_FLAGS) -o $@

executable/sparse: executable/pbit_coloring_cugraph_sparse
	cp $< $@

executable/dense: executable/pbit_coloring_cugraph_dense
	cp $< $@

.PRECIOUS: executable/graph_%.h


#*
#* User configurable simulation inputs
#


# GPU p-bit simulation
# 

# Circuit inputs
J ?= pbit-graphs/J_sparse.npy
H ?= pbit-graphs/h_sparse.npy
INITIAL_STATE ?= random

# Number of complete sweeps
NT ?= 1000

# Annealing settings
BETA_SCHEDULE ?= linear
BETA_START ?= 0.1
BETA_END ?= 5.0

# Random seed
SEED ?= 1234

# Output file
OUTPUT ?= results/annealing/final_states.txt


# Prepare user-selected circuit
.PHONY: prepare_sim simulate help_sim

prepare_sim:
	@mkdir -p executable
	$(PYTHON) scripts/prepare_sim.py \
		--J "$(J)" \
		--h "$(H)" \
		--initial-state "$(INITIAL_STATE)" \
		--nt "$(NT)" \
		--beta-schedule "$(BETA_SCHEDULE)" \
		--beta-start "$(BETA_START)" \
		--beta-end "$(BETA_END)" \
		--seed "$(SEED)" \
		--output executable/simulation_config.h


# Compile and run the GPU simulation
simulate: prepare_sim
	$(NVCC) -O3 -std=c++17 \
		-I executable \
		cuda/pbit_multiple_colors_test.cu \
		-o executable/pbit_simulation

	@mkdir -p "$(dir $(OUTPUT))"

	./executable/pbit_simulation > "$(OUTPUT)"

	@echo "Simulation complete."
	@echo "Final states saved to $(OUTPUT)"