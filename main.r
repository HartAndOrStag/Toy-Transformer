# sorry for over-commenting
# i am mostly just trying to explain it to myself

##### construct data set #####

# sequence of 3 consecutive numbers input
x <- matrix(data = NA, nrow = 21, ncol = 3)
for (i in 1:21) {
  x[i, ] <- c(i, i + 1, i + 2)
}

# output is next number in sequence
y <- matrix(4:24, nrow = 21, ncol = 1)


##### transformer algorithm #####

# set parameters
learning_rate <- 0.01
iterations <- 1000
hidden_dimensions <- 4

# initialize weights for attention layer
# These matrices are 1 x 3 to reflect that input sequence is 3 tokens long.
# It is divided by sqrt(3) to normalize for size of input dimension.
query_weight <- matrix(rnorm(3), nrow = 1, ncol = 3) / sqrt(3)
key_weight   <- matrix(rnorm(3), nrow = 1, ncol = 3) / sqrt(3)
value_weight <- matrix(rnorm(3), nrow = 1, ncol = 3) / sqrt(3)

# initialize weights for dense layers
# Here we have two dense layers. The first expands the input into a higher
# dimensional space. The second reduces it back to the original size.
# It is important to reduce the output size because later it will be added
# to the output of the attention layer. Again, each weight is divided by
# sqrt(input size) to normalize for size of input.
weight1 <- matrix(
  rnorm(3 * hidden_dimensions),
  nrow = 3,
  ncol = hidden_dimensions
) / sqrt(3)
bias1   <- rep(0, hidden_dimensions)
weight2 <- matrix(
  rnorm(3 * hidden_dimensions),
  nrow = hidden_dimensions,
  ncol = 3
) / sqrt(hidden_dimensions)
bias2   <- rep(0, 3)

# training loop
# This can be broken into two big chunks: the forward pass and
# the backward pass.
for (i in 1:iterations) {

  ##### forward pass #####

  # set up containers
  # Backpropagating through the attention layer is complicated and so
  # we need to set up tracking for intemediate values in the forword pass.
  # attention_output is a matrix the stores the output of the attention layer
  # for each row. caches is a list that stores various intermediate values
  # needed for the backward pass for each row.
  attention_output <- matrix(0, nrow = nrow(x), ncol = 3)
  caches <- vector("list", nrow(x))

  # attention forward pass algorithm
  # The attention algorithm is designed to determine how much attention each
  # token in a sequence should get. It outputs a set of attention weights
  # which correspond to probabilities over the sequence.
  for (row in seq_len(nrow(x))) {

    # isolate rows to process individually
    x_row <- x[row, ]
    tokens <- x_row

    # multiply each row by corresponding weight matrices
    query <- tokens %*% query_weight
    key   <- tokens %*% key_weight
    value <- tokens %*% value_weight

    # calculate attention weights
    # Query is called query because it is like it is asking a question.
    # It is compared to the key to determine the similarity score. This
    # is to be interpreted as how much each token attends to another token.
    # This is calcuated as the query multiplied by the transposed key,
    # divided by sqrt(input dimension of the key) to normalize. Dividing the
    # exponential scores matrix by the sum of the exponential score is called
    # the softmax function and it converts the scores into probabilities.
    scores <- query %*% t(key) / sqrt(ncol(key))
    exp_scores <- exp(scores)
    attention_weights <- exp_scores / rowSums(exp_scores)

    # calculate attention output
    # The attention weights are multiplied by the value matrix to convert
    # the weights back into actual values that can be propagated through
    # the model. The output needs to be converted back to a 1 x 3 matrix.
    # How this is done is a stylistic choice, but here we add the attention
    # output to the original tokens before pooling by averaging the columns.
    # Sweep is needed here because tokens is a vector.
    attention_output_row <- attention_weights %*% value
    residual <- sweep(attention_output_row, MARGIN = 1, tokens, FUN = "+")
    out <- colMeans(residual)

    # for each row, store output and cache intermediates
    attention_output[row, ] <- out
    caches[[row]] <- list(tokens = tokens,
                          query = query,
                          key = key,
                          value = value,
                          attention_weights = attention_weights,
                          attention_output = attention_output_row)
  }

  # dense forward pass algorithm
  # The dense layers are simply calculated as a linear transformation.
  # Negative values of the first dense layer are set to zero. This is
  # called ReLU activation and is there to introduce non-linearity, which
  # ensures the model can learn more than simple linear relationships.
  # Sweep is needed here because bias1 is a vector.
  dense1_input  <- sweep(
    attention_output %*% weight1,
    MARGIN = 2,
    bias1,
    FUN = "+"
  )
  dense1_output <- pmax(dense1_input, 0) # ReLU activation
  dim(dense1_output) <- dim(dense1_input)

  # The output of the second dense layer is added to the attention output.
  # This is called a residual connection and is there to preserve contextual
  # information through the layers. Sweep is needed here because bias2 is
  # a vector.
  dense2_input  <- sweep(
    dense1_output %*% weight2,
    MARGIN = 2,
    bias2,
    FUN = "+"
  )
  dense2_output <- dense2_input + attention_output

  # make prediction
  # The predicted value is just the average of the three output tokens.
  predictions <- rowMeans(dense2_output)

  # calculate loss
  # The loss function is the mean squared error.
  error <- predictions - y
  loss <- mean(error^2)


  ##### backward pass #####

  # dense layers backward pass algorithm
  # Here we go backwards through the model calculating gradients. Direction
  # of the gradients is determined by derivitves of the loss functions as
  # these relate to slope. As per the chain rule, each derivative is multiplied
  # by the gradient of the next layer. Each step is laid out seperately here.

  # This is the derivative of the mean squared error loss function.
  dpredictions <- (2 / nrow(y)) * error

  # The derivative of a mean is just each of the original number of values
  # split evenly.
  ddense2_output <- dpredictions %*% matrix(rep(1 / 3, 3), nrow = 1)

  # The derivatives of addition is just 1 for each input, so they pass
  # through unchanged at this step.
  dattention_output <- 1 * ddense2_output
  ddense2_input <- 1 * ddense2_output

  # calculate dense2 gradients
  # The derivative of a linear function is the coefficient, i.e. dense1_output.
  # Since dense1_output was originally multiplied by the weight2 matrix,
  # dense1_output must be transposed to recover the original shape of the
  # weight matrix.
  weight2_gradient <- t(dense1_output) %*% ddense2_input

  # The derivative of the bias is 1. The input is summed across columns because
  # the bias is added to all training examples.
  bias2_gradient <- 1 * colSums(ddense2_input)

  # Notice this is also the linear part of dense2 layer except this time the
  # derivative is taken with respect to dense1_output. Again,
  # the derivative of a linear function is the coefficient i.e. weight2.
  # Since dense1_output was originally multiplied by the weight2 matrix, weight2
  # must be transposed to recover the original shape of the dense output matrix.
  ddense1_output <- ddense2_input %*% t(weight2)

  # The derivative of ReLU is 1 for positive values and 0 for negative values.
  ddense1_input <- 1 * ddense1_output
  ddense1_input[dense1_input < 0] <- 0

  # calculate dense1 gradients
  # Since attention_output was originally multiplied by the weight1 matrix,
  # attention_output must be transposed to recover the shape of the original
  # matrix.
  weight1_gradient <- t(attention_output) %*% ddense1_input
  # The derivative of the bias is 1. The input is summed across columns because
  # the bias is added to all training examples.
  bias1_gradient   <- colSums(ddense1_input)

  # This is calculated the same as ddense1_output to propagate back to the
  # attention layer. Note that this is also added to the dattention_output that
  # came from the residual connection. This is because backpropagation requires
  # summing all contributions from multiple paths.
  dattention_output <- dattention_output + ddense1_input %*% t(weight1)

  # set up attention gradient matrices
  query_gradient <- matrix(0, nrow = 1, ncol = 3)
  key_gradient   <- matrix(0, nrow = 1, ncol = 3)
  value_gradient <- matrix(0, nrow = 1, ncol = 3)

  # attention backward pass algorithm
  for (row in seq_len(nrow(x))) {
    # The derivative of an assignment is 1, so it passes through unchanged.
    # do this per row
    dout <- 1 * dattention_output[row, ]

    # retrieve cached intermediates
    cache <- caches[[row]]

    query <- cache$query
    key   <- cache$key
    value <- cache$value
    attention_weights <- cache$attention_weights
    tokens <- cache$tokens
    key_dimensions <- ncol(key)

    # The derivative of a mean is just each of the original number of values
    # split evenly. The mean was taken for 3 values for each of 3 columns,
    # meaning it needs to be converted back to a 3 x 3 matrix.
    dattention_output_row <- matrix(
      rep(dout / 3, each = 3),
      nrow = 3,
      ncol = 3,
      byrow = TRUE
    )

    # The derivative of a linear function is the coefficient i.e. value.
    # Since attention_weights was originally multiplied by the value,
    # value must be transposed to recover the shape of the original matrix.
    dattention_weights <- dattention_output_row %*% t(value)
    # Since attention_weights was originally multiplied by value,
    # attention_weights must be transposed to recover the shape of
    # the original matrix.
    dvalue <- t(attention_weights) %*% dattention_output_row

    # The derivative of the softmax function involves many values so a
    # jacobian matrix is used to handle them cleanly. This jacobian matrix
    # has two parts. The diagonal part represents the derivative of the
    # softmax function with respect to its own score i=i which is equal to
    # p_i-p_i^2. The off-diagonal part represents the derivative of the softmax
    # function with respect to a different score i!=j which equals -p_i*p_j.
    # By placing the probabilities diagonally and subtracting by probabilities
    # multiplied by its transposition, we get p_i-p_i^2 on diagonal spaces.
    # Likewise, since off-diagonal spaces are zero, subtracting by the
    # probabilities multiplied by the transposition of the probabilities
    # gives -p_i*p_j on off-diagonal spaces.
    dscores <- matrix(0, nrow = 3, ncol = 3)
    for (j in 1:3) {
      probabilities <- attention_weights[j, ]
      jacobian_matrix <- diag(probabilities) -
        probabilities %*% t(probabilities)
      dscores[j, ] <- jacobian_matrix %*% dattention_weights[j, ]
    }

    # Since query was originally multiplied by the transposition of key,
    # key must be untransposed to recover the shape of the original matrix.
    # The 1 / sqrt(key_dimensions) is a constant
    dquery <- (dscores %*% key) / sqrt(key_dimensions)
    # Since query was originally multiplied by the transposition of key,
    # query must be transposed to recover the shape of the original matrix.
    dkey <- (t(query) %*% dscores) / sqrt(key_dimensions)

    # Query, key, and value all came from multiplying the tokens by the
    # corresponding weight. Thus, tokens must be transposed to recover
    # the original shape of each matrix.
    dquery_weight <- t(tokens) %*% dquery
    dkey_weight   <- t(tokens) %*% dkey
    dvalue_weight <- t(tokens) %*% dvalue

    # accumulate gradients for attention weights
    query_gradient <- query_gradient + dquery_weight
    key_gradient   <- key_gradient + dkey_weight
    value_gradient <- value_gradient + dvalue_weight
  }

  # update weights based on gradients
  weight2 <- weight2 - learning_rate * weight2_gradient
  bias2   <- bias2   - learning_rate * bias2_gradient
  weight1 <- weight1 - learning_rate * weight1_gradient
  bias1   <- bias1   - learning_rate * bias1_gradient

  query_weight <- query_weight - learning_rate * query_gradient
  key_weight   <- key_weight   - learning_rate * key_gradient
  value_weight <- value_weight - learning_rate * value_gradient
}


##### test predictions #####

print(cbind(
  input = apply(x, 1, paste, collapse = " "),
  target = y,
  prediction = round(predictions)
))