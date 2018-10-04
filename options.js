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
  // run some tests
  var en_check = "7543b6a357a043d7494b91942360a309aa2405b049c98f788763205401298765";
  var m_check = "inspire buffalo potato quantum aerobic two cement impulse neither brand churn battle pelican actress scorpion decrease month service sugar doll divorce network budget resource";
  
  if (entropyToMnemonic(en_check, enwords) !== m_check)
    alert("Failed to verify `entropyToMnemonic`");
  
  if (mnemonicToEntropy(m_check, enwords) !== en_check)
    alert("Failed to verify `mnemonicToEntropy`");
    
  var mnemonic = '';
  
  chrome.storage.sync.get('mnemonic', function(result) {
    if (typeof(result.mnemonic) == 'undefined' || result.mnemonic.length == 0) {
      $('#message').addClass('error').text(err);
      return;
    } else {
      mnemonic = result.mnemonic;
    }          
    var entropy = mnemonicToEntropy(mnemonic, enwords);
    $('#mnemonic').val(mnemonic).addClass('disabled');
    $('#entropy').val(entropy).addClass('disabled');
  });
  
  $('#regenerate').click(function(e) {
    mnemonic = generateMnemonic(128, randomHexArray, enwords);
    chrome.storage.sync.set({'mnemonic': mnemonic});
    
    $('#message').html("A new mnemonic was generated");
    location.reload(true);
  });
  $('#clear').click(function(e) {
    chrome.storage.sync.set({'mnemonic': ''});
    $('#message').html("Mnemonic cleared");
  });
  $('#save').click(function(e) {
    chrome.storage.sync.set({'mnemonic': $('#mnemonic').val()});
    $('#message').html("Mnemonic saved");
  });
});
