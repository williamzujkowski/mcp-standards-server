/**
 * User service following standards
 * @module UserService
 */

class UserService {
  constructor(database, logger) {
    this.db = database;
    this.logger = logger;
  }

  /**
   * Get user by ID
   * @param {string} userId - The user ID
   * @returns {Promise<User>} The user object
   * @throws {Error} If user not found
   */
  async getUser(userId) {
    try {
      this.logger.info(`Fetching user: ${userId}`);
      
      const user = await this.db.users.findById(userId);
      
      if (!user) {
        throw new Error(`User not found: ${userId}`);
      }
      
      return this.sanitizeUser(user);
    } catch (error) {
      this.logger.error(`Failed to get user: ${error.message}`);
      throw error;
    }
  }

  /**
   * Sanitize user data for response
   * @private
   */
  sanitizeUser(user) {
    const { password, ...safeUser } = user;
    return safeUser;
  }
}

module.exports = UserService;
