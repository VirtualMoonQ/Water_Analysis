# ==============================
# PHẦN 3: TIỀN XỬ LÝ DỮ LIỆU
# ==============================

library(tidyverse)
library(caret)
library(corrplot)
library(randomForest)

# Read data
df <- read.csv("water_potability.csv")

cat("=== STRUCTURE ===\n")
str(df)

cat("\n=== SUMMARY ===\n")
summary(df)

# Missing values (before)
cat("\n=== MISSING VALUES (BEFORE) ===\n")
print(colSums(is.na(df)))

# Replace NA with median
for (col in names(df)) {
  df[is.na(df[[col]]), col] <- median(df[[col]], na.rm = TRUE)
}

# Missing values (after)
cat("\n=== MISSING VALUES (AFTER) ===\n")
print(colSums(is.na(df)))

# Normalization
preproc <- preProcess(df[, -10], method = c("range"))
X_scaled <- predict(preproc, df[, -10])
df_scaled <- cbind(X_scaled, Potability = df$Potability)

cat("\n=== DATA AFTER NORMALIZATION ===\n")
print(head(df_scaled))

# ==============================
# PHẦN 4: THỐNG KÊ MÔ TẢ
# ==============================

# Save histogram
png("histograms.png", width=1200, height=800)

df %>%
  pivot_longer(cols = -Potability, names_to = "variable", values_to = "value") %>%
  ggplot(aes(x = value)) +
  geom_histogram(bins = 30, fill = "blue") +
  facet_wrap(~variable, scales = "free")

dev.off()

# Save potability distribution
png("potability.png", width=600, height=400)

ggplot(df, aes(x = as.factor(Potability))) +
  geom_bar(fill = "orange") +
  labs(x = "Potability", y = "Count", title = "Distribution of Potability")

dev.off()

# Save correlation plot
png("correlation.png", width=800, height=800)

cor_matrix <- cor(df)
corrplot(cor_matrix, method = "color", tl.cex = 0.7)

dev.off()

# ==============================
# BOXPLOT (SO SÁNH THEO POTABILITY)
# ==============================

library(gridExtra)

p1 <- ggplot(df, aes(x = as.factor(Potability), y = Organic_carbon, fill = as.factor(Potability))) +
  geom_boxplot() +
  labs(title = "Organic Carbon by Potability")

p2 <- ggplot(df, aes(x = as.factor(Potability), y = Trihalomethanes, fill = as.factor(Potability))) +
  geom_boxplot() +
  labs(title = "Trihalomethanes by Potability")

p3 <- ggplot(df, aes(x = as.factor(Potability), y = Turbidity, fill = as.factor(Potability))) +
  geom_boxplot() +
  labs(title = "Turbidity by Potability")

# Save boxplot
png("boxplots.png", width=1200, height=400)

grid.arrange(p1, p2, p3, ncol = 3)

dev.off()

# ==============================
# BOXPLOT - GROUP 2
# ==============================

p4 <- ggplot(df, aes(x = as.factor(Potability), y = Chloramines, fill = as.factor(Potability))) +
  geom_boxplot() +
  labs(title = "Chloramines by Potability")

p5 <- ggplot(df, aes(x = as.factor(Potability), y = Sulfate, fill = as.factor(Potability))) +
  geom_boxplot() +
  labs(title = "Sulfate by Potability")

p6 <- ggplot(df, aes(x = as.factor(Potability), y = Conductivity, fill = as.factor(Potability))) +
  geom_boxplot() +
  labs(title = "Conductivity by Potability")

png("boxplots_group2.png", width=1200, height=400)
grid.arrange(p4, p5, p6, ncol = 3)
dev.off()


# ==============================
# BOXPLOT - GROUP 3
# ==============================

p7 <- ggplot(df, aes(x = as.factor(Potability), y = ph, fill = as.factor(Potability))) +
  geom_boxplot() +
  labs(title = "pH by Potability")

p8 <- ggplot(df, aes(x = as.factor(Potability), y = Hardness, fill = as.factor(Potability))) +
  geom_boxplot() +
  labs(title = "Hardness by Potability")

p9 <- ggplot(df, aes(x = as.factor(Potability), y = Solids, fill = as.factor(Potability))) +
  geom_boxplot() +
  labs(title = "Solids by Potability")

png("boxplots_group3.png", width=1200, height=400)
grid.arrange(p7, p8, p9, ncol = 3)
dev.off()

# ==============================
# PHẦN 5: MODEL
# ==============================

set.seed(42)

trainIndex <- createDataPartition(df_scaled$Potability, p = 0.8, list = FALSE)

train_data <- df_scaled[trainIndex, ]
test_data  <- df_scaled[-trainIndex, ]

# Logistic Regression
model_logit <- glm(Potability ~ ., data = train_data, family = "binomial")

pred_prob <- predict(model_logit, test_data, type = "response")
pred_logit <- ifelse(pred_prob > 0.5, 1, 0)

cat("\n=== LOGISTIC REGRESSION ===\n")
cm_logit <- confusionMatrix(
  factor(pred_logit, levels = c(0,1)),
  factor(test_data$Potability, levels = c(0,1))
)
print(cm_logit)

# Random Forest
model_rf <- randomForest(as.factor(Potability) ~ ., data = train_data)

pred_rf <- predict(model_rf, test_data)

cat("\n=== RANDOM FOREST ===\n")
cm_rf <- confusionMatrix(
  factor(pred_rf, levels = c(0,1)),
  factor(test_data$Potability, levels = c(0,1))
)
print(cm_rf)

# Print accuracy clearly
cat("\n=== ACCURACY COMPARISON ===\n")
cat("Logistic Regression:", cm_logit$overall['Accuracy'], "\n")
cat("Random Forest:", cm_rf$overall['Accuracy'], "\n")