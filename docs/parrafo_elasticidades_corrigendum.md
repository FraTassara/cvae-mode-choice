# Paragraph for the corrigendum: elasticities (Table 9)

The elasticities reported in the published Table 9 were computed on a single cross-validation partition. Recomputed as means over the 50 partitions (Table 9, corrected), the own-elasticity of car travel time under CVAE is −0.088 (SD 0.131) and positive in only 30% of partitions, so the published value of +0.014 reflects one partition rather than a systematic pattern; the baseline value also changes (−0.146 rather than −0.180). The reversal that CVAE produces systematically in SM-Centro concerns waiting time: the own-elasticity of waiting time becomes positive in 100% of partitions for both Metro (+0.116, SD 0.020) and Bus (+0.062, SD 0.011), with the corresponding cross-elasticities negative in every partition — the full inverted pattern implied by a positive waiting-time coefficient. Under all other methods the waiting-time elasticity is negative and stable (positive in at most 6% of partitions), although the balancing methods attenuate it relative to the unbalanced baseline (from −0.119 to between −0.052 and −0.084 for Metro). For travel time, CVAE does not reverse the sign on average but makes the estimate markedly less stable (SD 0.131 versus 0.054–0.076 under the other methods), a distinct form of behavioral degradation. The claim that a substantial proportion of individuals show increasing choice probability of the car as travel time increases is therefore withdrawn; the corresponding systematic effect is in waiting time.

---

*Trazabilidad (no va al paper).* Texto redactado por la autora. Cada número
se verificó contra `results/corrigendum/efectos_marginales_resumen_sm_centro.csv`
(50 particiones, `08_efectos_marginales.R`) y `tablas_paper/tabla_09.csv`:

- CVAE, TVIA1 (CD): −0.088, SD 0.131, positiva en el 30 %.
- Original, TVIA1: −0.146; el publicado era −0.180 (`docs/valores_paper.csv`).
- CVAE, espera propia: TESP4 (Metro) +0.116, SD 0.020; TESP5 (Bus) +0.062,
  SD 0.011; ambas positivas en el 100 %.
- CVAE, elasticidades cruzadas de la espera (TESP4 → CD y Bus; TESP5 → CD y
  Metro): positivas en el 0 % de las particiones, es decir, negativas en todas.
- Otros métodos, espera propia: positiva en a lo sumo el 6 % (ROS). Metro:
  original −0.119, ROS −0.052, class-weighting −0.056, SMOTE −0.084.
- Otros métodos, SD de TVIA1: 0.054 (ROS) a 0.076 (original).
