library(amcat4r)  # if needed: remotes::install_github("ccs-amsterdam/amcat4r")
library(boolydict)# if needed: remotes::install_github("kasperwelbers/boolydict")
library(tidyverse)

dict <- read_csv("data/raw/Social_groups_dictionary.csv") |>
  pivot_longer(everything(), values_to = "query") |>
  na.omit() |>
  filter(str_ends(name, "D")) |>
  mutate(query=str_remove_all(query, "^\\* "),
         query=if_else(str_detect(query, " "), str_c('"', query, '"'), query)) |>
  group_by(name) |>
  summarize(label=head(query, 1), query=str_c(query, collapse=" OR "))


amcat_login("https://open.amcat.nl/amcat")

done = c()

for (var in unique(dict$name)) {
  q=dict |> filter(name==var)
  if (q$label %in% done) next
  message(q$label)
  r <- amcat4r::query_documents("2023-nl-manifestos", queries=q$query, fields=c(".id"), per_page=100, max_page=Inf, scroll="5m")
  message(str_c(q$label, " -> ", nrow(r), " hits"))
  if (nrow(r) > 0) amcat4r::update_tags("2023-nl-manifestos", action = "add", tag = q$label, field = "group",ids = r$.id)
  done = c(done, q$label)
  Sys.sleep(1)
}

