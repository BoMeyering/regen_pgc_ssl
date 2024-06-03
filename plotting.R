# library(jsonlite)
# library(tidyverse)
# 
# img_counts = fromJSON('metadata/class_img_counts.json')
# px_counts = fromJSON('metadata/class_pixel_counts.json')
# 
# class_indices = names(img_counts)
# class_names = c(
#   'Background', 
#   'Quadrat',
#   'PGC Grass', 
#   'PGC Clover', 
#   'Broadleaf Weed',
#   'Maize', 
#   'Soybean', 
#   'Other Vegetation'
# )
# 
# img_counts = unname(unlist(img_counts))
# px_counts = unname(unlist(px_counts))
# 
# df = data.frame(index=class_indices, class=class_names, img_counts=img_counts, px_counts=px_counts)
# 
# img_p = ggplot(df, aes(x = reorder(class, img_counts), y = img_counts, fill=class))+
#   geom_bar(stat='identity')+
#   scale_fill_viridis_d(option="D", begin=.1, end=.9)+
#   coord_flip()+
#   labs(x = 'Label Class', y = 'Image Counts', fill = 'Class')+
#   theme_minimal()
# 
# img_p
# 
# px_p = ggplot(df, aes(x = reorder(class, px_counts), y = px_counts, fill=class))+
#   geom_bar(stat='identity')+
#   scale_fill_viridis_d(option="D", begin=.1, end=.9)+
#   coord_flip()+
#   labs(x = 'Label Class', y = 'Pixel Counts', fill = 'Class')+
#   theme_minimal()
# 
# px_p
# 
# px_log_p = ggplot(df, aes(x = reorder(class, px_counts), y = px_counts, fill=class))+
#   geom_bar(stat='identity')+
#   scale_fill_viridis_d(option="D", begin=.1, end=.9)+
#   coord_flip()+
#   labs(x = 'Label Class', y = 'Pixel Counts', fill = 'Class')+
#   theme_minimal()+
#   scale_y_log10()
# 
# px_log_p




library(jsonlite)
library(tidyverse)

# Load the data
img_counts = fromJSON('metadata/class_img_counts.json')
px_counts = fromJSON('metadata/class_pixel_counts.json')

# Define class names
class_indices = names(img_counts)
class_names = c(
  'Background', 
  'Quadrat',
  'PGC Grass', 
  'PGC Clover', 
  'Broadleaf Weed',
  'Maize', 
  'Soybean', 
  'Other Vegetation'
)

# Prepare the data
img_counts = unname(unlist(img_counts))
px_counts = unname(unlist(px_counts))

df = data.frame(index=class_indices, class=class_names, img_counts=img_counts, px_counts=px_counts)

# Define a common theme for consistency
common_theme <- theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, size = 12),
    axis.text.y = element_text(size = 12),
    axis.title = element_text(size = 14, face = "bold"),
    plot.title = element_text(size = 16, face = "bold", hjust = 0.5),
    legend.position = "none"
  )

# Image counts plot
img_p = ggplot(df, aes(x = reorder(class, img_counts), y = img_counts, fill=class)) +
  geom_bar(stat='identity') +
  scale_fill_viridis_d(option="D", begin=.1, end=.9) +
  coord_flip() +
  labs(
    title = 'Image Counts by Label Class',
    x = 'Label Class', 
    y = 'Image Counts'
  ) +
  common_theme

# Display the plot
img_p

# Pixel counts plot
px_p = ggplot(df, aes(x = reorder(class, px_counts), y = px_counts, fill=class)) +
  geom_bar(stat='identity') +
  scale_fill_viridis_d(option="D", begin=.1, end=.9) +
  coord_flip() +
  labs(
    title = 'Pixel Counts by Label Class',
    x = 'Label Class', 
    y = 'Pixel Counts'
  ) +
  common_theme

# Display the plot
px_p

# Log-transformed pixel counts plot
px_log_p = ggplot(df, aes(x = reorder(class, px_counts), y = px_counts, fill=class)) +
  geom_bar(stat='identity') +
  scale_fill_viridis_d(option="D", begin=.1, end=.9) +
  coord_flip() +
  scale_y_log10() +
  labs(
    title = 'Log-Transformed Pixel Counts by Label Class',
    x = 'Label Class', 
    y = 'Pixel Counts (log scale)'
  ) +
  common_theme

# Display the plot
px_log_p
