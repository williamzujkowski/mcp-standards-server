// no jsdoc
function getUser(id) {
  var user = db.users.find(function(u) { return u.id == id });  // == instead of ===
  return user;  // returns password too
}

// callback hell
function updateUser(id, data, callback) {
  db.users.find(id, function(err, user) {
    if (err) callback(err);
    else {
      db.users.update(id, data, function(err2, result) {
        if (err2) callback(err2);
        else {
          db.logs.add('updated', function(err3) {
            callback(err3, result);
          });
        }
      });
    }
  });
}

eval("console.log('unsafe')");  // security issue
