library(readtext)
library(here)
library(tidytext)
library(tidyverse)

d_raw <- readtext(here("data/raw/*.pdf"))
# d |> select(title) |> unique() |> write_csv(here("data/raw/meta.csv"))
meta <- read_csv(here("data/raw/meta.csv")) |> rename(file=title)

d <- d_raw |> 
  unnest_tokens(paragraph, text, token="paragraphs", to_lower=FALSE) |> 
  mutate(paragraph=str_replace_all(paragraph, "\\s+", " ") |> trimws())|> 
  group_by(doc_id, paragraph) |> filter(n() <= 2) |> ungroup() |>
  filter(str_detect(paragraph, "[A-Za-z]")) |> 
  select(file=doc_id, text=paragraph) |> 
  add_column(date="2023-10-01") |>
  inner_join(meta) |>
  mutate(title=str_c(party, " partijprogramma 2023")) |>
  select(-file)


View(d)

library(amcat4r)
#amcat4r::amcat_login("https://amcat4.labs.vu.nl/amcat")
#amcat4r::delete_index("2023-nl-manifestos")


amcat4r::amcat_login("https://open.amcat.nl/amcat")
amcat4r::delete_index("2023-nl-manifestos")
amcat4r::create_index("2023-nl-manifestos", name="2023 Dutch Manifestos")
amcat4r::set_fields("2023-nl-manifestos", list(party="keyword"))
amcat4r::upload_documents("2023-nl-manifestos", d)



