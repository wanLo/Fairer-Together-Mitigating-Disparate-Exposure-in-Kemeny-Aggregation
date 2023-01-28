
library(cowplot)
library(tidyverse)
library(readr)
library("gridExtra")
library(ggpubr)
library(ggtext)
library(ggthemes)
library(penngradlings)# for text font
results <- read_csv("mallows_results.csv")
results$consensus_accuracy <- signif(results$consensus_accuracy, digits = 2)
results$exposure_ratio <- signif(results$exposure_ratio, digits = 2)



gamma_r <- read_csv("mallows_gamma_results.csv")
gamma_r$consensus_accuracy <- signif(gamma_r$consensus_accuracy, digits = 2)
gamma_r$exposure_ratio <- signif(gamma_r$exposure_ratio, digits = 2)


mallows_string <- "mallows dispersion (\U1D719)"
rr_string <- "reference ranking"
high_consensus_color <- "#88419d"#"#3f007d"
high_fair_color <- "#31a354"
consensus_limits <- c(.4,1)
exposure_limits <- c(.4,1)
fairness_legend <- "ER"
consensus_legend <- "CA"
textsize <- 3

consensus_map <- function(dataframe, dataset) {
  

  data <-  dataframe %>%
    filter(.data$method== .env$dataset)
  plot <- ggplot(data, aes(mallows_dispersion, as.factor(reference_ranking))) +
  geom_tile(aes(fill = consensus_accuracy), colour = "white") +
  scale_fill_gradient(low = "white", high = high_consensus_color, limits = consensus_limits, name = consensus_legend)+
  geom_text(aes(label = consensus_accuracy), size = textsize, color = "black")+
  labs(title = glue::glue({dataset}))+theme_tufte()+
  theme(text=element_text(family="Times New Roman"))+
  theme(legend.position = "right",
        legend.direction = "vertical")+
  scale_y_discrete(name = rr_string)+
  scale_x_continuous(name = mallows_string,breaks=c(0,.2,.4,.6,.8,1))
  
  return(plot)
  
}


fairness_map <- function(dataframe, dataset) {
  
  
  data <-  dataframe %>%
    filter(.data$method== .env$dataset)
  plot <- ggplot(data, aes(mallows_dispersion, as.factor(reference_ranking))) +
      geom_tile(aes(fill = exposure_ratio), colour = "white") +
      scale_fill_gradient(low = "white", high = high_fair_color, limits = exposure_limits, name = fairness_legend)+
      geom_text(aes(label = exposure_ratio), size = textsize, color = "black")+
      labs(title =  glue::glue({dataset}))+theme_tufte()+
      theme(text=element_text(family="Times New Roman"))+
      theme(legend.position = "right",
            legend.direction = "vertical")+
      scale_y_discrete(name = rr_string)+
      scale_x_continuous(name = mallows_string,breaks=c(0,.2,.4,.6,.8,1))
  
  return(plot)
  
}


c_kem <- consensus_map(results, "KEMENY")
f_kem <- fairness_map(results, "KEMENY")

c_epik <- consensus_map(results, "EPIK")
f_epik <- fairness_map(results, "EPIK")

c_rapf <- consensus_map(results, "RAPF")
f_rapf <- fairness_map(results, "RAPF")

c_pre <- consensus_map(results, "PRE-FE")
f_pre <- fairness_map(results, "PRE-FE")

c_fkem <- consensus_map(results, "FAIRKEMENY")
f_fkem <- fairness_map(results, "FAIRKEMENY")


c_epira_kem <- consensus_map(results, "EPIRA+Kemeny")
f_epira_kem <- fairness_map(results, "EPIRA+Kemeny")

c_epira_cop <- consensus_map(results, "EPIRA+Copeland")
f_epira_cop <- fairness_map(results, "EPIRA+Copeland")

c_epira_sch <- consensus_map(results, "EPIRA+Schulze")
f_epira_sch <- fairness_map(results, "EPIRA+Schulze")

c_epira_bord <- consensus_map(results, "EPIRA+Borda")
f_epira_bord <- fairness_map(results, "EPIRA+Borda")

c_epira_maxi <- consensus_map(results, "EPIRA+Maximin")
f_epira_max <- fairness_map(results, "EPIRA+Maximin")



figure_C_kem <- ggarrange(c_kem, c_epik,
                      ncol = 2, nrow = 1, common.legend = TRUE, legend = "right")

figure_f_kem <- ggarrange(f_kem, f_epik,
                      ncol = 2, nrow = 1, common.legend = TRUE, legend = "right")

comp_kem <- ggarrange(figure_C_kem, figure_f_kem, ncol = 2, nrow = 1)

# pdfwidth <- 15
# pdfheight <- 3 
pdfwidth <- 12
pdfheight <- 2
#uncomment
# ggsave(comp_kem, filename = 'plots/heatmaps_kem_epik.pdf', device = cairo_pdf,
#        width = pdfwidth, height = pdfheight, units = "in")




figure_C <- ggarrange(c_epira_cop, c_fkem, c_rapf, c_pre,
                      ncol = 2, nrow = 2, common.legend = TRUE, legend = "right")

figure_f <- ggarrange(f_epira_cop, f_fkem, f_rapf, f_pre,
                      ncol = 2, nrow = 2, common.legend = TRUE, legend = "right")

comp <- ggarrange(figure_C, figure_f, ncol = 2, nrow = 1)
pdfwidth <- 12
pdfheight <- 4

#uncomment
# ggsave(comp, filename = 'plots/comparative_maps.pdf', device = cairo_pdf,
#        width = pdfwidth, height = pdfheight, units = "in")





## ABLATION STUDY

ablation <- read_csv("mallows_results.csv") %>%
  select(method, mallows_dispersion, reference_ranking, consensus_accuracy) %>%
  filter(method %in% c("EPIRA+Copeland_noWiG", "EPIRA+Copeland"))%>%
  pivot_wider(names_from = method, values_from = consensus_accuracy, values_fill = 0)

ablation$delta <- ablation$`EPIRA+Copeland` - ablation$`EPIRA+Copeland_noWiG`
ablation$percentage_increase <- (ablation$`EPIRA+Copeland` - ablation$`EPIRA+Copeland_noWiG`)/ablation$`EPIRA+Copeland_noWiG`

mallows_string <- "mallows dispersion (\U1D719)"
rr_string <- "reference ranking"
low_color <- "#fef0d9"
high_color <- "#d7301f"
fill_limits <- c(0, 4)
fairness_legend <- "ER"
consensus_legend <- "% Increase"
textsize <- 3

ablation$increase <- ablation$percentage_increase*100 #percentage

ablation$increase <- signif(ablation$increase, digits = 2)

a_plot <- ggplot(ablation, aes(mallows_dispersion, as.factor(reference_ranking))) +
  geom_tile(aes(fill = increase), colour = "white") +
  scale_fill_gradient(low = "white", high = high_color, limits = fill_limits, name = consensus_legend)+
  geom_text(aes(label = increase), size = textsize, color = "black")+
  labs(title = "% increase in Consensus Accuracy (CA) of EPIRA")+theme_tufte()+
  theme(text=element_text(family="Times New Roman"))+
  theme(legend.position = "right",
        legend.direction = "vertical")+
  scale_y_discrete(name = rr_string)+
  scale_x_continuous(name = mallows_string,breaks=c(0,.2,.4,.6,.8,1))


pdfwidth <- 5
pdfheight <- 3
#uncomment
# ggsave(a_plot, filename = 'plots/epir_ablation.pdf', device = cairo_pdf,
#        width = pdfwidth, height = pdfheight, units = "in")


#GAMMA VALUES

consensus_legend <- "CA"
six <- gamma_r %>%
  filter(Gamma == 0.6)


c_epik6 <- consensus_map(six, "EPIK")
f_epik6 <- fairness_map(six, "EPIK")
c_epira_cop6 <- consensus_map(six, "EPIRA+Copeland")
f_epira_cop6 <- fairness_map(six, "EPIRA+Copeland")
figure_C_6 <- ggarrange(c_epik6, c_epira_cop6,
                          ncol = 2, nrow = 1, common.legend = TRUE, legend = "right")

figure_f_6 <- ggarrange(f_epik6, f_epira_cop6,
                          ncol = 2, nrow = 1, common.legend = TRUE, legend = "right")

comp_6 <- ggarrange(figure_C_6, figure_f_6, ncol = 2, nrow = 1)

pdfwidth <- 12
pdfheight <- 2
#uncomment
ggsave(comp_6, filename = 'plots/gamma_six.pdf', device = cairo_pdf,
       width = pdfwidth, height = pdfheight, units = "in")


seven <- gamma_r %>%
  filter(Gamma == 0.7)


c_epik7 <- consensus_map(seven, "EPIK")
f_epik7 <- fairness_map(seven, "EPIK")
c_epira_cop7 <- consensus_map(seven, "EPIRA+Copeland")
f_epira_cop7 <- fairness_map(seven, "EPIRA+Copeland")
figure_C_7 <- ggarrange(c_epik7, c_epira_cop7,
                        ncol = 2, nrow = 1, common.legend = TRUE, legend = "right")

figure_f_7 <- ggarrange(f_epik7, f_epira_cop7,
                        ncol = 2, nrow = 1, common.legend = TRUE, legend = "right")

comp_7 <- ggarrange(figure_C_7, figure_f_7, ncol = 2, nrow = 1)


#uncomment
ggsave(comp_7, filename = 'plots/gamma_seven.pdf', device = cairo_pdf,
       width = pdfwidth, height = pdfheight, units = "in")


eight <- gamma_r %>%
  filter(Gamma == 0.8)


c_epik8 <- consensus_map(eight, "EPIK")
f_epik8 <- fairness_map(eight, "EPIK")
c_epira_cop8 <- consensus_map(eight, "EPIRA+Copeland")
f_epira_cop8 <- fairness_map(eight, "EPIRA+Copeland")
figure_C_8 <- ggarrange(c_epik8, c_epira_cop8,
                        ncol = 2, nrow = 1, common.legend = TRUE, legend = "right")

figure_f_8 <- ggarrange(f_epik8, f_epira_cop8,
                        ncol = 2, nrow = 1, common.legend = TRUE, legend = "right")

comp_8 <- ggarrange(figure_C_8, figure_f_8, ncol = 2, nrow = 1)


#uncomment
ggsave(comp_8, filename = 'plots/gamma_eight.pdf', device = cairo_pdf,
       width = pdfwidth, height = pdfheight, units = "in")






