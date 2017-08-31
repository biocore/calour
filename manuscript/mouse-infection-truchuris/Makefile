pre ?= all
seq ?= $(pre).seqs.fa
base = $(basename $(seq))
out ?= $(base).aln
aln ?= $(out)/$(base)_aligned.fasta
fil_dir ?= $(out)/$(base)_aligned_filtered/
fil ?= $(fil_dir)/$(base)_aligned_pfiltered.fasta
tre ?= $(base).tre

align: $(aln)
$(aln): $(seq)
	align_seqs.py -i $(seq) -o $(out) --template_fp ~/database/core_set_aligned.fasta.imputed

filter: $(fil)
$(fil): $(aln)
	filter_alignment.py -i $(aln) -o $(fil_dir)

tree: $(tre)
$(tre): $(fil)
	make_phylogeny.py -i $(fil) -o $(tre)


taxa:
	assign_taxonomy.py -i $(seq) -m uclust
	biom add-metadata -i $(pre).biom -o $(pre).tax.biom --observation-metadata-fp uclust_assigned_taxonomy/$(base)_tax_assignments.txt --sc-separated taxonomy --observation-header OTUID,taxonomy
	# assign_taxonomy.py -i $(seq) -r unite_01.08.2015/developer/sh_refs_qiime_ver7_97_01.08.2015_dev.fasta -t unite_01.08.2015/developer/sh_taxonomy_qiime_ver7_97_01.08.2015_dev.txt -m blast
