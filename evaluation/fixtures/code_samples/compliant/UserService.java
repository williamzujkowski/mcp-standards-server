package com.example.service;

import java.util.Optional;
import java.util.logging.Logger;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

/**
 * Service for managing user operations.
 */
@Service
public class UserService {
    
    private static final Logger LOGGER = Logger.getLogger(UserService.class.getName());
    
    private final UserRepository userRepository;
    private final PasswordEncoder passwordEncoder;
    
    /**
     * Constructs a new UserService.
     * 
     * @param userRepository the user repository
     * @param passwordEncoder the password encoder
     */
    public UserService(UserRepository userRepository, PasswordEncoder passwordEncoder) {
        this.userRepository = userRepository;
        this.passwordEncoder = passwordEncoder;
    }
    
    /**
     * Retrieves a user by ID.
     * 
     * @param userId the user ID
     * @return the user if found
     * @throws UserNotFoundException if the user is not found
     */
    @Transactional(readOnly = true)
    public User getUser(Long userId) {
        LOGGER.info("Fetching user with ID: " + userId);
        
        return userRepository.findById(userId)
            .orElseThrow(() -> new UserNotFoundException("User not found: " + userId));
    }
    
    /**
     * Creates a new user.
     * 
     * @param userDto the user data
     * @return the created user
     */
    @Transactional
    public User createUser(CreateUserDto userDto) {
        validateUserDto(userDto);
        
        User user = new User();
        user.setEmail(userDto.getEmail());
        user.setName(userDto.getName());
        user.setPassword(passwordEncoder.encode(userDto.getPassword()));
        
        return userRepository.save(user);
    }
    
    private void validateUserDto(CreateUserDto userDto) {
        if (userDto.getEmail() == null || userDto.getEmail().isEmpty()) {
            throw new IllegalArgumentException("Email cannot be empty");
        }
        // Additional validation...
    }
}
