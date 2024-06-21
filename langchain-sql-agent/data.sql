-- Insert data into address table
INSERT INTO address (street, city, state, postcode, created_at) VALUES
('123 Main St', 'Redfern', 'NSW', '2016', CURRENT_TIMESTAMP),
('456 Elm St', 'Strathfield', 'NSW', '2135', CURRENT_TIMESTAMP),
('789 Oak St', 'Chatswood', 'NSW', '2067', CURRENT_TIMESTAMP);

-- Insert data into users table
INSERT INTO users (email, firstName, lastName, address_id, created_at) VALUES
('john.doe@example.com', 'John', 'Doe', 1, CURRENT_TIMESTAMP),
('jane.smith@example.com', 'Jane', 'Smith', 2, CURRENT_TIMESTAMP),
('bob.johnson@example.com', 'Bob', 'Johnson', 3, CURRENT_TIMESTAMP);

-- Insert data into claim table
INSERT INTO claim (amount, description, type, status, user_id, created_at) VALUES
(100, 'floods', 'FLOODS', 'Pending', 1, CURRENT_TIMESTAMP),
(250, 'bush filre', 'BUSH_FIRE', 'Approved', 2, CURRENT_TIMESTAMP),
(75, 'hail damage', 'STORM_DAMAGE', 'Denied', 3, CURRENT_TIMESTAMP),
(150, 'land slide', 'LAND_SLIDE', 'Pending', 1, CURRENT_TIMESTAMP),
(300, 'floods', 'FLOODS', 'Approved', 2, CURRENT_TIMESTAMP);