require("@nomicfoundation/hardhat-toolbox");

module.exports = {
  solidity: "0.8.19",
  networks: {
    hardhat: {},
    testnet: {
      url: "https://data-seed-prebsc-1-s1.binance.org:8545",
      accounts: []
    }
  }
};
