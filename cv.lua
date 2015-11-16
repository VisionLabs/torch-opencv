-- By default, package.path doesn't have './?/init.lua'
-- As a temporary workaround, this file will be removed
-- as soon as we have a rockspec file.

package.path = './?/init.lua;' .. package.path
return dofile 'cv/init.lua'
