x <- 5
y = 10

num_var <- 5.7
char_var <- "Hello, R!"
logical_var <- TRUE


numeric_vector <- c(1, 2, 3, 4, 5)
char_vector <- c("apple", "banana", "orange")

data <- data.frame(
  Name = c("Alice", "Bob", "Charlie"),
  Age = c(25, 30, 22),
  Score = c(95, 80, 90)
)

square <- function(x) {
  return(x^2)
}

result <- square(4)



for (i in 1:5) {
  print(i)
}

if (x > 0) {
  print("Positive")
} else {
  print("Non-positive")
}



x <- c(1, 2, 3, 4, 5)
y <- c(2, 4, 6, 8, 10)
plot(x, y, type = "l", col = "blue", lwd = 2)


prices <- tq_get("AAPL",
  get = "stock.prices",
  from = "2000-01-01",
  to = "2022-12-31"
)