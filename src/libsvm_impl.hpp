/**
 * libsvm_impl.hpp
 *
 * Trivially simple loader for libsvm data.
 */
#ifndef LIBSVM_IMPL_HPP
#define LIBSVM_IMPL_HPP

#include "libsvm.hpp"

/**
 * Given a filename, return a matrix containing the data and the labels.
 * The size is inferred from the dataset.
 * No comment lines are allowed.
 */
template<typename MatType>
typename std::tuple<MatType, arma::rowvec>
load_libsvm(const std::string& filename)
{
  arma::wall_clock c;
  c.tic();

  // For the first pass, we have to compute the size of the matrix.
  std::fstream f(filename);
  if (!f.good())
  {
    std::ostringstream oss;
    oss << "Error opening file '" << filename << "' for reading." << std::endl;
    throw std::runtime_error(oss.str());
  }

  size_t max_row = 0;
  std::string line;
  size_t line_num = 1;
  while (std::getline(f, line))
  {
    // Right-trim string.
    line.erase(line.find_last_not_of(" \n\r\t") + 1);

    // For the first pass, we only need the last dimension (we are assuming that
    // dimensions are given in increasing order).  If that assumption is wrong,
    // the second pass will crash.
    const size_t last_space = line.rfind(' ');
    const size_t last_colon = line.rfind(':');
    if (last_space != std::string::npos && last_colon != std::string::npos &&
        last_space < last_colon)
    {
      // There might be no spaces if, e.g., the row is completely empty and has
      // no label.
      const size_t dim_len = last_colon - last_space - 1;
      const int raw_dim = atoi(line.substr(last_space + 1, dim_len).c_str());
      if (raw_dim <= 0)
      {
        std::ostringstream oss;
        oss << "Error on line " << line_num << ": could not extract dimension "
            << "from final token " << line.substr(last_space + 1) << " (note, "
            << "dimensions must start from 1)";
        throw std::runtime_error(oss.str());
      }

      const size_t adj_dim = raw_dim - 1;
      max_row = std::max(max_row, adj_dim);
    }
    else if (last_space != std::string::npos &&
             last_colon != std::string::npos &&
             last_space >= last_colon)
    {
      std::ostringstream oss;
      oss << "Error on line " << line_num << " extracting dimension from last "
          << "token";
      throw std::runtime_error(oss.str());
    }

    ++line_num;
  }
  --line_num; // Remove extra increment.

  const double first_pass_time = c.toc();
  std::cout << "File '" << filename << "' contains a matrix with " << line_num
      << " observations in " << max_row + 1 << " dimensions." << std::endl;
  std::cout << "First pass took " << first_pass_time << "s." << std::endl;

  c.tic();
  MatType result(max_row + 1, line_num);
  arma::rowvec labels(line_num);

  f.close();
  f.open(filename);
  if (!f.good())
  {
    std::ostringstream oss;
    oss << "Error opening file '" << filename << "' for reading." << std::endl;
    throw std::runtime_error(oss.str());
  }

  line_num = 0; // not offset by one this time
  char** tokens = new char*[result.n_rows + 1];
  size_t* dims = new size_t[result.n_rows + 1];
  double* vals = new double[result.n_rows + 1];

  while (std::getline(f, line))
  {
    if (f.fail())
    {
      std::ostringstream oss;
      oss << "Error on line " << line_num << ": failed to read line";
      throw std::runtime_error(oss.str());
    }

    // Right-trim string.
    line.erase(line.find_last_not_of(" \n\r\t") + 1);

    // First, tokenize the entire line.
    char* c_line = new char[line.length() + 1];
    c_line[line.length()] = '\0';
    strncpy(c_line, line.c_str(), line.length());

    size_t token_count = 1;
    tokens[0] = c_line;
    for (size_t i = 0; i < line.length(); i++)
    {
      if (c_line[i] == ' ')
      {
        c_line[i] = '\0';
        tokens[token_count] = &(c_line[i + 1]);
        ++token_count;
      }
    }

    // Handle label separately. Enforce label in {-1, +1}
    float label_i = atof(tokens[0]);
    labels[line_num] = (label_i == 1.0) ? 1.0 : -1.0;

    if (token_count <= 1)
    {
      delete[] c_line;
      continue; // Only a label for this point---all zeros.
    }

    #pragma omp parallel for num_threads(4)
    for (size_t token = 1; token < token_count; ++token)
    {
      // Each token should be "dim:val".
      // We'll start by finding the colon.
      char* colon = tokens[token];
      while (*colon != ':' && *colon != '\0')
        ++colon;

      if (*colon == '\0')
      {
        std::ostringstream oss;
        oss << "Error on line " << line_num + 1 << ": no ':' found in token: "
            << tokens[token];
        throw std::runtime_error(oss.str());
      }
      else if (colon == tokens[token])
      {
        std::ostringstream oss;
        oss << "Error on line " << line_num + 1 << ": no dimension found for "
            << "token: " << tokens[token];
        throw std::runtime_error(oss.str());
      }

      *colon = '\0'; // Now we have two separate tokens.
      char* val_str = (colon + 1);

      // Read the dimension.
      const int dim = atoi(tokens[token]);
      if (dim == 0)
      {
        std::ostringstream oss;
        oss << "Error on line " << line_num + 1 << ": could not parse dimension"
            << ": " << tokens[token] << "; note that dimensions must start from"
            << " 1";
        throw std::runtime_error(oss.str());
      }
      const size_t dim_adj = dim - 1;

      // Now read the value.
      errno = 0;
      char** strtod_out = 0;
      const double val = std::strtod(val_str, strtod_out);
      if (errno == ERANGE)
      {
        std::ostringstream oss;
        oss << "Error on line " << line_num + 1 << ": value '" << val_str << "'"
            << " is out of range";
        throw std::runtime_error(oss.str());
      }
      else if (strtod_out == &val_str)
      {
        std::ostringstream oss;
        oss << "Error on line " << line_num + 1 << ": value '" << val_str << "'"
            << " could not be parsed to floating-point";
        throw std::runtime_error(oss.str());
      }

      dims[token - 1] = dim_adj;
      vals[token - 1] = val;
    }

    delete[] c_line;

    // Now serially insert all tokens into the matrix.
    for (size_t i = 0; i < token_count - 1; ++i)
    {
      result(dims[i], line_num) = vals[i];
    }

    ++line_num;
  }

  delete[] tokens;
  delete[] dims;
  delete[] vals;

  const double load_time = c.toc();
  std::cout << "Second pass for loading took " << load_time << "s."
      << std::endl;

  return std::make_tuple(result, labels);
}

#endif
