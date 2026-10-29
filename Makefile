# Build every algorithm/graph combination with: make -j2 all
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

.PHONY: all sparse dense
all: $(VERSIONS)

# Retain the previous short cuGraph executable names.
sparse: executable/sparse
dense: executable/dense

executable/graph_%.h: pbit-graphs/J_%.npy pbit-graphs/h_%.npy scripts/prepare_graph.py
	$(PYTHON) scripts/prepare_graph.py --J pbit-graphs/J_$*.npy \
		--h pbit-graphs/h_$*.npy --output $@

# Each CPU algorithm shares the original main() and selects one function.
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
