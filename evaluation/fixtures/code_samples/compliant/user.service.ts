/**
 * User service for managing user operations
 */

import { Injectable, Logger } from '@nestjs/common';
import { User, CreateUserDto, UpdateUserDto } from './user.types';
import { UserRepository } from './user.repository';
import { UserNotFoundError, ValidationError } from '../errors';

@Injectable()
export class UserService {
  private readonly logger = new Logger(UserService.name);

  constructor(
    private readonly userRepository: UserRepository,
  ) {}

  /**
   * Get a user by ID
   * @param userId - The user's ID
   * @returns The user object
   * @throws {UserNotFoundError} If user is not found
   */
  async getUser(userId: string): Promise<User> {
    this.logger.log(`Fetching user: ${userId}`);
    
    const user = await this.userRepository.findById(userId);
    
    if (!user) {
      throw new UserNotFoundError(`User not found: ${userId}`);
    }
    
    return this.sanitizeUser(user);
  }

  /**
   * Create a new user
   * @param createUserDto - The user creation data
   * @returns The created user
   */
  async createUser(createUserDto: CreateUserDto): Promise<User> {
    this.validateCreateUserDto(createUserDto);
    
    const hashedPassword = await this.hashPassword(createUserDto.password);
    
    const user = await this.userRepository.create({
      ...createUserDto,
      password: hashedPassword,
    });
    
    return this.sanitizeUser(user);
  }

  /**
   * Remove sensitive data from user object
   */
  private sanitizeUser(user: User): User {
    const { password, ...sanitized } = user;
    return sanitized as User;
  }

  /**
   * Validate user creation data
   */
  private validateCreateUserDto(dto: CreateUserDto): void {
    if (!dto.email || !this.isValidEmail(dto.email)) {
      throw new ValidationError('Invalid email address');
    }
    
    if (!dto.password || dto.password.length < 8) {
      throw new ValidationError('Password must be at least 8 characters');
    }
  }

  private isValidEmail(email: string): boolean {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
  }

  private async hashPassword(password: string): Promise<string> {
    // Implementation would use bcrypt or similar
    return `hashed_${password}`;
  }
}
