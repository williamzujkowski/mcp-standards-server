package main

import (
    "github.com/gin-gonic/gin"
    "github.com/sirupsen/logrus"
)

func main() {
    logger := logrus.New()
    logger.Info("Starting microservice")
    
    router := gin.Default()
    
    router.GET("/health", func(c *gin.Context) {
        c.JSON(200, gin.H{"status": "healthy"})
    })
    
    router.Run(":8080")
}
