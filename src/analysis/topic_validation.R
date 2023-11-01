library(tidyverse)
ds = read_csv("data/intermediate/topics_validation.csv") |> 
  mutate(id=seq_along(chunkid),
         relevant=ifelse(relevant == "LABEL_0", "relevant", "irrelevant") |> as.factor())

# compute majority vote, drop sentences without majority
manual <- ds |> select(id, manual1, manual2, manual3) |> 
  pivot_longer(-id) |>
  group_by(id, value) |> 
  summarize(n=n()) |>
  mutate(max=max(n), tot=n()) |>
  filter(!(max==1 & tot==3), n==max) |>
  select(id, manual=value) |> 
  ungroup()



d <- inner_join(ds, manual) |>
  mutate(relevance_manual = ifelse(manual == "000", "irrelevant", "relevant") |> as.factor(),
         manual =as.factor(manual),
         topic1=str_remove_all(topic1, " - .*"),
         topic2=str_remove_all(topic2, " - .*"),
         topic3=str_remove_all(topic3, " - .*"),
         )

levels = union(d$topic1, d$topic2) |> union(d$topic3) |> union(d$manual)
d <- d |> mutate(topic1=factor(topic1, levels=levels),
                 topic2=factor(topic2, levels=levels),
                 topic3=factor(topic3, levels=levels),
                manual=factor(manual, levels=levels),
            )
d

levels(d$relevance_manual)
# accuracy of relevance classifier
bind_rows(
yardstick::precision(d, truth=relevance_manual, estimate=relevant, event_level="first"),
yardstick::recall(d, truth=relevance_manual, estimate=relevant, event_level="first"),
yardstick::f_meas(d, truth=relevance_manual, estimate=relevant, event_level="first"),
yardstick::accuracy(d, truth=relevance_manual, estimate=relevant, event_level="first"),
) 

# headline performance of classifier  
# (suppress warnings about levels not existing in predictions)
suppressWarnings(bind_rows(
  yardstick::precision(d, truth=manual, estimate=topic1),
  yardstick::recall(d, truth=manual, estimate=topic1),
  yardstick::f_meas(d, truth=manual, estimate=topic1),
  yardstick::accuracy(d, truth=manual, estimate=topic1),
))

# headline performance of classifier (taking only relevant classes)
d_rel = filter(d, relevance_manual == "relevant")
suppressWarnings(bind_rows(
  yardstick::precision(d_rel, truth=manual, estimate=topic1),
  yardstick::recall(d_rel, truth=manual, estimate=topic1),
  yardstick::f_meas(d_rel, truth=manual, estimate=topic1),
  yardstick::accuracy(d_rel, truth=manual, estimate=topic1),
))

# How often is manual class in top 3?
d_rel |> mutate(intop3 = manual == topic1 | manual == topic2 | manual == topic3) |> pull(intop3) |> mean()
d_rel |> mutate(intop2 = manual == topic1 | manual == topic2) |> pull(intop2) |> mean()
d_rel |> mutate(iscorrect = manual == topic1) |> pull(iscorrect) |> mean()


cmp_vu = read_csv("data/raw/cmp_topics.csv") |> mutate(cmp=factor(cmp), label=factor(label))

d_rel <- d_rel |> inner_join(select(cmp_vu, topic1=cmp, topic1_lbl=label)) |>
  inner_join(select(cmp_vu, topic2=cmp, topic2_lbl=label)) |>
  inner_join(select(cmp_vu, topic3=cmp, topic3_lbl=label)) |>
  inner_join(select(cmp_vu, manual=cmp, manual_lbl=label))

tops = tribble(
  ~.metric, ~.estimate, 
  "top3", d_rel |> mutate(intop3 = manual_lbl == topic1_lbl | manual_lbl == topic2_lbl | manual_lbl == topic3_lbl) |> pull(intop3) |> mean(),
  "top2", d_rel |> mutate(intop2 = manual_lbl == topic1_lbl | manual_lbl == topic2_lbl) |> pull(intop2) |> mean(),
  "top1", d_rel |> mutate(iscorrect = manual_lbl == topic1_lbl) |> pull(iscorrect) |> mean())

suppressWarnings(bind_rows(
  yardstick::precision(d_rel, truth=manual_lbl, estimate=topic1_lbl),
  yardstick::recall(d_rel, truth=manual_lbl, estimate=topic1_lbl),
  yardstick::f_meas(d_rel, truth=manual_lbl, estimate=topic1_lbl),
  yardstick::accuracy(d_rel, truth=manual_lbl, estimate=topic1_lbl),
)) |> select(-.estimator)  |> bind_rows(tops)
