'use strict';

// Make sure we are in strict mode.
(function() {
  var strictMode = false;
  try {
    NaN = NaN;
  } catch (err) {
    strictMode = true;
  }
  if (!strictMode) {
    throw 'Unable to activate strict mode.';
  }
})();


$(function() {
  // Get the current tab.
  chrome.tabs.query({
      active: true,
      currentWindow: true
    }, function(tabs) {
      var showError = function(err) {
        $('#domain').val('N/A').addClass('disabled');
        $('#domain').prop('disabled', true);
        $('#key').prop('disabled', true);
        $('#hash').prop('disabled', true);
        $('p:not(#message)').addClass('disabled');
        $('#message').addClass('error').text(err);
      };

      // Make sure we got the tab.
      if (tabs.length !== 1) {
        return showError('Unable to determine active tab.');
      }

      // Get the domain.
      var domain = null;
      var username = null;
      var matches = tabs[0].url.match(/^http(?:s?):\/\/([^\.]*?\.)?([^\.]*\.[^/]*)/);
      if (matches) {
        domain = matches[2].toLowerCase();
      } else {
        // Example cause: files served over the file:// protocol.
        return showError('Unable to determine the domain.');
      }
      if (/^http(?:s?):\/\/chrome\.google\.com\/webstore.*/.test(tabs[0].url)) {
        // Technical reason: Chrome prevents content scripts from running in the app gallery.
        return showError('Try Hashpass on another domain.');
      }
      $('#domain').val(domain);
      
      var mnemonic = '';
      var entropy = '';
      var domain_key = domain.replace(".", "_").toLowerCase();;
      
      chrome.storage.sync.get(null, function(result) {
        if (typeof(result.mnemonic) == 'undefined' || result.mnemonic.length == 0) {
          chrome.runtime.openOptionsPage();
          return showError('The mnemonic must be generated.');
        } else {
          mnemonic = result.mnemonic;
        }
        
        username = result[domain_key];
        console.log(result);
        
        // Run the content script to register the message handler.
        chrome.tabs.executeScript(tabs[0].id, {
          file: 'content_script.js'
        }, function() {
          entropy = mnemonicToEntropy(mnemonic, enwords);
          mnemonic = '';
        
          // Check if a password field is selected.
          chrome.tabs.sendMessage(tabs[0].id, {
              type: 'hashpassCheckIfPasswordField'
            }, function(response) {
              
              if (typeof(username) == 'undefined') {
                chrome.tabs.sendMessage(tabs[0].id, {
                    type: 'getUserName'
                  }, function(response) {
                    if (response.username.length > 0) {
                      $('#user').val(response.username);
                      $('#key').focus();
                      username = response.username;
                      console.log(username);
                    } else {
                      // Focus the text field.
                      $('#user').focus();
                    }
                  }
                );
              } else {
                $('#user').val(username);
                $('#key').focus();
              }
              
              // Different user interfaces depending on whether a password field is in focus.
              var passwordMode = (response.type === 'password');
              if (passwordMode) {
                $('#message').html('Press <strong>ENTER</strong> to fill in the password field.');
                // $('#hash').val('[hidden]').addClass('disabled');
              } else {
                $('#message').html('<strong>TIP:</strong> Select a password field first.');
              }
        
              // Called whenever the key changes.
              var update = function() {
                // Compute the first 16 base64 characters of iterated-SHA-256(domain + '/' + key, 2 ^ difficulty).
                var key = $('#key').val();
                username = $('#user').val();

                var pwd = generateFoldedPassword(entropy, key, domain, username, '0');
                
                if (!passwordMode)
                {
                  $('#hash').val(pwd);
                }

                return pwd;
              };
        
              // A debounced version of update().
              var timeout = null;
              var debouncedUpdate = function() {
                if (timeout !== null) {
                  clearInterval(timeout);
                }
                timeout = setTimeout((function() {
                  update();
                  timeout = null;
                }), 100);
              };
        
              if (passwordMode) {
                // Listen for the Enter key.
                $('#domain, #key, #user').keydown(function(e) {
                  if (e.which === 13) {
                    var obj = {};
                    obj[domain_key] = username;
                    chrome.storage.sync.set(obj);

                    // Try to fill the selected password field with the hash.
                    chrome.tabs.sendMessage(tabs[0].id, {
                        type: 'hashpassFillPasswordField',
                        hash: update()
                      }, function(response) {
                        // If successful, close the popup.
                        if (response.type === 'close') {
                          window.close();
                        }
                      }
                    );
                  }
                });
              }
        
              if (!passwordMode) 
              {
                // Register the update handler.
                $('#domain, #key, #user').bind('propertychange change keyup input paste', debouncedUpdate);
        
                // Update the hash right away.
                debouncedUpdate();
              }
            }
          );
        });
      });
    }
  );
});
