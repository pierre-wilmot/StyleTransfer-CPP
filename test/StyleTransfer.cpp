#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include "catch.hpp"
#include "StyleTransfer.h"

TEST_CASE( "Test Output Size", "[StyleTransfer]" )
{
  StyleTransfer model;
  torch::Tensor t = torch::rand({1, 3, 128, 128});
  torch::Tensor res = model->forward(t);
  REQUIRE( res.sizes() == c10::IntArrayRef({1, 512, 8, 8}) );
}
