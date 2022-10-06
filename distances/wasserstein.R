suppressPackageStartupMessages({require(MASS, quietly=TRUE)
require(mvtnorm, quietly=TRUE)
library(GSAR, quietly=TRUE)
library(devtools, quietly=TRUE)
# install_github("gmordant/WassersteinGoF", ref = "main")
library(WassersteinGoF, quietly=TRUE)})
options(warn=-1)

WassersteinTest <- function(sample, ref) {
  return(WassersteinDist(sample, ref))
}
