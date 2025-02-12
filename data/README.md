1. Datasets S1-S7, protein quantifications in all samples, and protein database.

Datasets S1-S7 and protein quantifications are .csv files.

- Data S1. (Data_S1_Bbigelowii_proteins.csv): Details of principal B. bigelowii localized proteins.
- Data S2. (Data_S2_UCYN-A_proteins.csv): Details of principal UCYN-A encoded proteins.
- Data S3. (Data_S3_Biosynthesis.csv): Details of biosynthesis related proteins
- Data S4. (Data_S4_Diazotrophy_related.csv): Details of diazotrophy and antioxidant related proteins
- Data S5. (Data_S5_Ferredoxin_and_flavodox.csv): Details of B. bigelowii and UCYN-A encoded ferredoxins and flavodoxins.
- Data S6. (Data_S6_Regulatory_etc.csv): Details of regulatory proteins
- Data S7. (Data_S7_Fig3Ainfo.xlsx): Details and abbreviations of proteins mentioned in Fig. 3A.
- Column headings for S1-S7: detected in proteomes? = Yes if protein was detected in any proteomic sample; uTP? = Yes if uTP sequence was detected using HMM approach; Enriched in whole-cell or UCYN-A samples, DESeq2 Wald test, P<0.05 = identifies sample type in which protein is more abundant; Log2FC UCYN-A vs B. bigelowii samples (positive = UCYN-A) = gives log base 2 fold change between sample types with positive values indicating increased abundance in UCYN-A samples; Average quantity = average protein quantity (area-under-curve of the mass spectrometric peaks of all assigned peptides) in all triplicate samples.

2. ADK1075_proteomics_DB_2.fasta fasta file containing all predicted B. bigelowii and UCYN-A proteins. These are generated from UCYN-A2 genome (NCBI RefSeq GCF 020885515.1), B. bigelowii transcriptome (DDBJ/Genbank/ENA DRA011134), and B. bigelowii plastid genome (Genbank OR912953, OR912954 and OR912955)

3. ADK1075_ProteinQuantifications.csv contains protein quantification data for all proteins in all samples. Log2 fold change (positive = UCYN-A) and P values are calculated between UCYN-A and whole culture samples at day and night using DESeq2 Wald test. The Sequence source column gives the origin of the sequence as either previously published B. bigelowii transcriptomes, previously published UCYN-A2 genome, B. bigelowii chloroplast genome (sequenced for this work), or contaminants.

4. UCYNA2_genome_description.csv gives functional annotations of the UCYN-A2 genome from 10.3389/fpls.2021.749895

5. Bbigelowii_transcriptome_annotations.csv gives functional annotations of the B. bigelowii transcriptomes from 10.3389/fpls.2021.749895

6. uTP_HMM_hits.fasta fasta file containing B. bigelowii protein sequences retrieved with our HMM search of ADK1075_proteomics_DB_2.fasta

7. Import_candidates.fasta fasta file containing all B. bigelowii protein sequences predicted to localize to UCYN-A, either by HMM detection of uTP, significant enrichment in protein, or both
